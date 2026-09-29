FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image ~1 GB instead of ~5 GB
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml README.md ./
COPY pxai ./pxai
RUN pip install ".[app,api]" pydicom

COPY app.py ./
COPY api ./api
COPY configs ./configs

# Bake the weights into the image so containers start offline
RUN python -c "from pxai.model import download_weights; download_weights()"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000
# Default: Streamlit demo. For the API:
#   docker run -p 8000:8000 pxai uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
