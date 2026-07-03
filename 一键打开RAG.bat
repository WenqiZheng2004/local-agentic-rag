@echo off
chcp 65001 >nul
title Local Agentic RAG

echo Starting Local Agentic RAG...
echo.
echo A browser tab will open at http://127.0.0.1:7860
echo Keep this window open while using the app.
echo.

start "" cmd /c "timeout /t 10 /nobreak >nul && start http://127.0.0.1:7860"

wsl.exe -e bash -lc "cd ~/local-agentic-rag && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ai && printf '%s\n' 'LLM_MODEL=Qwen/Qwen2.5-3B-Instruct' 'EMBED_MODEL=BAAI/bge-small-zh-v1.5' 'LOAD_IN_4BIT=true' 'DEVICE=cuda' 'USE_RERANKER=false' 'RETRIEVE_K=8' 'TOP_K=4' 'MAX_REWRITES=1' > .env && python -u app.py"

echo.
echo RAG app stopped. You can close this window.
pause
