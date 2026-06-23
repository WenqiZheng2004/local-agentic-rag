"""Agentic RAG built on LangGraph.

Instead of a fixed "retrieve -> stuff -> answer" pipeline, the agent reasons
about each step and can self-correct:

    START
      |
      v
   [route] --- direct ---------------------------------> [generate]  (empty KB)
      |
   retrieve
      v
  [condense]  (rewrite a follow-up into a standalone query, using chat history)
      |
      v
  [retrieve] -> [rerank] -> [grade] --- relevant ------> [generate] --> END
                              |
                         not relevant
                              |
                      (rewrites < MAX_REWRITES)
                              v
                        [transform] -- new query --> [retrieve]   (loop once)

Three things make this "agentic" rather than a vanilla RAG, and they're the
parts worth talking about in an interview:
  1. condense  — history-aware retrieval (resolves "what about X?" follow-ups)
  2. rerank    — a cross-encoder re-scores candidates for precision
  3. grade + transform — the self-correcting retry loop
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from config import config
from .llm import LocalLLM
from .vectorstore import VectorStore


class AgentState(TypedDict, total=False):
    question: str                       # original user question (this turn)
    query: str                          # current search query (condensed / rewritten)
    history: List[Dict[str, str]]       # prior turns: [{"role","content"}, ...]
    route: str                          # "retrieve" | "direct"
    documents: List[Dict[str, Any]]
    relevant: bool
    rewrites: int
    sources: List[str]
    generation: str


def _yes(text: str) -> bool:
    t = text.strip().lower()
    return (t.startswith("yes") or "yes" in t.split()[:3]) if t else False


def _format_history(history: List[Dict[str, str]], max_turns: int = 4) -> str:
    if not history:
        return ""
    recent = history[-max_turns * 2:]  # a "turn" is a user+assistant pair
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


class RAGAgent:
    def __init__(self, llm: LocalLLM | None = None, store: VectorStore | None = None):
        self.llm = llm or LocalLLM()
        self.store = store or VectorStore()
        self.reranker = self._maybe_load_reranker()
        self.graph = self._build()

    def _maybe_load_reranker(self):
        """Load the cross-encoder if enabled; degrade gracefully on failure."""
        if not config.use_reranker:
            return None
        try:
            from .reranker import Reranker
            return Reranker()
        except Exception as e:
            print(f"[RAGAgent] Reranker unavailable ({e}); using vector order only.")
            return None

    # ----- graph wiring -----
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("route", self.route_node)
        g.add_node("condense", self.condense_node)
        g.add_node("retrieve", self.retrieve_node)
        g.add_node("rerank", self.rerank_node)
        g.add_node("grade", self.grade_node)
        g.add_node("transform", self.transform_node)
        g.add_node("generate", self.generate_node)

        g.add_edge(START, "route")
        g.add_conditional_edges(
            "route", lambda s: s["route"],
            {"retrieve": "condense", "direct": "generate"},
        )
        g.add_edge("condense", "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "grade")
        g.add_conditional_edges(
            "grade", self._after_grade,
            {"generate": "generate", "transform": "transform"},
        )
        g.add_edge("transform", "retrieve")
        g.add_edge("generate", END)
        return g.compile()

    # ----- nodes -----
    def route_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        # Empty knowledge base -> nothing to retrieve, answer directly.
        if self.store.count() == 0:
            return {"route": "direct", "query": question, "rewrites": 0}
        # KB has content -> always retrieve. A 3B model's "should I search?"
        # judgement is unreliable, so we skip it and let grading handle quality.
        return {"route": "retrieve", "query": question, "rewrites": 0}

    def condense_node(self, state: AgentState) -> AgentState:
        """Turn a context-dependent follow-up into a standalone search query.

        e.g. history mentions remote work, user asks "那试用期员工呢?" ->
        "试用期员工的远程办公政策是怎样的?". No history -> pass through.
        """
        history = state.get("history") or []
        if not history:
            return {"query": state["question"]}
        hist = _format_history(history)
        system = ("Given the chat history and a follow-up question, rewrite the "
                  "follow-up into a standalone question that makes sense on its "
                  "own. Output only the rewritten question, nothing else.")
        user = f"Chat history:\n{hist}\n\nFollow-up: {state['question']}\n\nStandalone question:"
        condensed = self.llm.complete(system, user, max_new_tokens=64, temperature=0.0).strip()
        return {"query": condensed or state["question"]}

    def retrieve_node(self, state: AgentState) -> AgentState:
        # Pull a wide candidate pool; the reranker will narrow it down.
        hits = self.store.query(state.get("query", state["question"]),
                                top_k=config.retrieve_k)
        return {"documents": hits}

    def rerank_node(self, state: AgentState) -> AgentState:
        docs = state.get("documents", [])
        query = state.get("query", state["question"])
        if self.reranker is not None and docs:
            docs = self.reranker.rerank(query, docs, top_k=config.top_k)
        else:
            docs = docs[:config.top_k]  # fall back to vector order
        return {"documents": docs}

    def grade_node(self, state: AgentState) -> AgentState:
        docs = state.get("documents", [])
        if not docs:
            return {"relevant": False}
        context = "\n---\n".join(d["text"][:300] for d in docs)
        system = ("You grade whether the retrieved context is relevant enough to "
                  "answer the question. Reply with only 'yes' or 'no'.")
        user = (f"Question: {state['question']}\n\nContext:\n{context}\n\n"
                "Is this context relevant enough to answer? yes or no.")
        ans = self.llm.complete(system, user, max_new_tokens=4, temperature=0.0)
        return {"relevant": _yes(ans)}

    def _after_grade(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("rewrites", 0) >= config.max_rewrites:
            return "generate"  # give up rewriting; answer with what we have
        return "transform"

    def transform_node(self, state: AgentState) -> AgentState:
        system = ("Rewrite the user's question into a more effective search query "
                  "for document retrieval. Output only the rewritten query.")
        user = f"Question: {state['question']}\nRewritten query:"
        new_q = self.llm.complete(system, user, max_new_tokens=64, temperature=0.0).strip()
        return {"query": new_q or state["question"],
                "rewrites": state.get("rewrites", 0) + 1}

    def generate_node(self, state: AgentState) -> AgentState:
        docs = state.get("documents", [])
        history = state.get("history") or []
        hist_block = _format_history(history)
        hist_prefix = f"Conversation so far:\n{hist_block}\n\n" if hist_block else ""

        if state.get("route") == "direct" or not docs:
            system = ("You are a helpful assistant. Answer concisely and in the "
                      "user's language. If you are unsure, say so.")
            ans = self.llm.complete(system, hist_prefix + state["question"])
            return {"generation": ans, "sources": []}

        parts, sources = [], []
        for i, d in enumerate(docs):
            src = d.get("metadata", {}).get("source", "unknown")
            parts.append(f"[{i + 1}] (source: {src})\n{d['text']}")
            sources.append(src)
        context = "\n\n".join(parts)

        system = ("Answer the question using ONLY the provided context. "
                  "Cite the supporting snippets inline as [1], [2], etc. "
                  "If the context does not contain the answer, say you cannot find "
                  "it in the documents. Answer in the user's language.")
        user = f"{hist_prefix}Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
        ans = self.llm.complete(system, user)

        seen = list(dict.fromkeys(sources))  # de-dupe, keep order
        return {"generation": ans, "sources": seen}

    # ----- public API -----
    def run(self, question: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        result = self.graph.invoke({
            "question": question,
            "history": history or [],
            "documents": [],
            "sources": [],
            "rewrites": 0,
        })
        return {
            "answer": result.get("generation", ""),
            "sources": result.get("sources", []),
            "documents": result.get("documents", []),
            "route": result.get("route", "direct"),
            "rewrites": result.get("rewrites", 0),
        }
