FROM python:3.11-slim
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/

COPY app.py .
EXPOSE 7860
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HF_HOME=/tmp/huggingface
CMD ["python", "app.py"]

