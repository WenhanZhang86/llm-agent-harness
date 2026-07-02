FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY agent_harness ./agent_harness
COPY providers ./providers
COPY configs ./configs
COPY evals ./evals

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "agent_harness.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
