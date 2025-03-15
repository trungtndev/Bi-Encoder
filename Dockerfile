### Pip stage ###

FROM python:3.11-slim as compiler
ENV PYTHONUNBUFFERED 1

ADD . .

RUN python -m venv /venv

ENV PATH="/venv/bin:$PATH"

RUN pip install nvidia-cuda-runtime-cu11 && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install fastapi[standard] uvicorn python-multipart && \
    pip install tritonclient[all] sentence-transformers underthesea

FROM python:3.11-slim as runner

COPY --from=compiler /venv /venv
COPY utils /utils
COPY api /api

ENV PATH="/venv/bin:$PATH"

CMD fastapi run api/fastapp.py --port 7999