FROM python:3.12-slim AS builder
WORKDIR /app

RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

# Pre-download HuggingFace models during build to avoid cold-start in demo
RUN python -c "from FlagEmbedding import FlagModel; FlagModel('BAAI/bge-m3', use_fp16=True)"
RUN python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)"

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY src/ src/
COPY data/ data/

ENV PYTHONPATH=/app/src
CMD ["uvicorn", "medical_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
