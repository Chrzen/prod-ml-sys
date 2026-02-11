FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd -m appuser
WORKDIR /app

# Copy both pyproject and source BEFORE installing
COPY services/inference_service/pyproject.toml ./pyproject.toml
COPY services/inference_service/src ./src

RUN pip install --upgrade pip \
    && pip install .

# Copy model artifact
COPY artifacts/run1/model.pkl ./artifacts/run1/model.pkl

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["serve-model"]
