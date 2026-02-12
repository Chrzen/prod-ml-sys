FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd -m appuser
WORKDIR /app

# Copy both pyproject and source BEFORE installing
COPY services/training_service/pyproject.toml ./pyproject.toml
COPY services/training_service/src ./src

RUN pip install .

RUN mkdir -p /app/artifacts
RUN chown -R appuser:appuser /app
USER appuser

CMD ["train-model"]
