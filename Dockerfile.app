FROM python:3.11-slim
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/

COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]

