FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY test/ test/
COPY pytest.ini .
COPY degradation_transformer_model.safetensors .
COPY degradation_transformer_model_config.json .
CMD ["pytest", "test"]

