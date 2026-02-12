FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd -m appuser
WORKDIR /app

COPY services/inference_service/pyproject.toml ./pyproject.toml
COPY services/inference_service/src ./src

RUN pip install --upgrade pip \
    && pip install .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["serve-model"]
