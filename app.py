"""Gradio UI for the local agentic RAG assistant.

Run from the project root:

    python app.py

Then open the printed local URL. Upload .pdf/.txt/.md files, click
"Index documents", and chat. Models load lazily on first use.

Features:
  - multi-turn chat (the agent sees prior turns and resolves follow-ups)
  - a "retrieved context" panel showing exactly which chunks were used
"""

from __future__ import annotations

import os
import warnings

# --- Quiet the noise so the console (and demo recordings) stay clean ---
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # chromadb
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import gradio as gr

from src.vectorstore import VectorStore
from src.llm import LocalLLM
from src.ingest import ingest_file
from src.agent import RAGAgent

# Lazy singleton so the UI starts instantly; the model loads on first request.
_STATE = {"agent": None}


def get_agent() -> RAGAgent:
    if _STATE["agent"] is None:
        store = VectorStore()
        llm = LocalLLM()
        _STATE["agent"] = RAGAgent(llm=llm, store=store)
    return _STATE["agent"]


def index_files(files):
    if not files:
        return "No files uploaded."
    agent = get_agent()
    lines, total = [], 0
    for f in files:
        path = f if isinstance(f, str) else f.name
        try:
            n = ingest_file(path, store=agent.store)
            total += n
            lines.append(f"✓ {os.path.basename(path)} — {n} chunks")
        except Exception as e:
            lines.append(f"✗ {os.path.basename(path)} — failed: {e}")
    lines.append(f"\nTotal chunks in store: {agent.store.count()}")
    return "\n".join(lines)


def _format_sources(out) -> str:
    docs = out.get("documents", [])
    if not docs:
        return f"_No documents retrieved (route = {out.get('route')})._"
    blocks = [f"**route = {out.get('route')}, rewrites = {out.get('rewrites')}**\n"]
    for i, d in enumerate(docs, 1):
        src = d.get("metadata", {}).get("source", "unknown")
        score = d.get("rerank_score")
        score_str = f" · rerank={score:.2f}" if score is not None else ""
        snippet = d["text"].strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        blocks.append(f"**[{i}] {src}{score_str}**\n{snippet}")
    return "\n\n".join(blocks)


def on_user(message, chat_history):
    """Append the user's message and clear the textbox."""
    if not message.strip():
        return "", chat_history
    chat_history = chat_history + [{"role": "user", "content": message}]
    return "", chat_history


def on_bot(chat_history):
    """Run the agent on the latest user message, with prior turns as history."""
    if not chat_history:
        return chat_history, ""
    user_msg = chat_history[-1]["content"]
    prior = chat_history[:-1]
    agent = get_agent()
    out = agent.run(user_msg, history=prior)
    chat_history = chat_history + [{"role": "assistant", "content": out["answer"]}]
    return chat_history, _format_sources(out)


with gr.Blocks(title="Local Agentic RAG") as demo:
    gr.Markdown(
        "# 🧠 Local Agentic RAG\n"
        "Upload your documents, then ask questions. Retrieval, reranking, "
        "reasoning and generation all run **locally on your GPU** — no API keys, "
        "no data leaves your machine."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(
                label="Upload .pdf / .txt / .md",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md"],
            )
            index_btn = gr.Button("Index documents", variant="primary")
            status = gr.Textbox(label="Index status", lines=6, interactive=False)
            index_btn.click(index_files, inputs=file_in, outputs=status)

            with gr.Accordion("🔎 Retrieved context (last answer)", open=False):
                sources_md = gr.Markdown("_Ask something to see what was retrieved._")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=480, label="Chat")
            msg = gr.Textbox(placeholder="问点什么…  e.g. 入职满5年有多少天年假?",
                             label="Your message", scale=4)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

    # Wiring: add user msg -> run agent -> update chat + sources panel.
    send.click(on_user, [msg, chatbot], [msg, chatbot]).then(
        on_bot, chatbot, [chatbot, sources_md])
    msg.submit(on_user, [msg, chatbot], [msg, chatbot]).then(
        on_bot, chatbot, [chatbot, sources_md])
    clear.click(lambda: ([], "_Cleared._"), None, [chatbot, sources_md])


if __name__ == "__main__":
    demo.launch()
