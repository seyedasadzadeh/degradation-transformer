FROM python:3.11-slim
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/

COPY app.py .
EXPOSE 7860
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HF_HOME=/tmp/huggingface
ENV WANDB_CACHE_DIR=/tmp/wandb
ENV WANDB_CONFIG_DIR=/tmp/wandb
ENV WANDB_DATA_DIR=/tmp/wandb_data
CMD ["python", "app.py"]

