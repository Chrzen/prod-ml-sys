# --- Base image ---
FROM python:3.9-slim

# --- System settings ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1    

# --- Create app user ---
RUN useradd -m appuser

# --- Set working directory ---
WORKDIR /app

# ---- Copy dependency metadata first (for caching) ----
COPY pyproject.toml .

# --- Install dependencies ---
RUN pip install --upgrade pip \
    && pip install .

# --- Copy application code ---
COPY src ./src

# --- Copy artifacts ---
COPY ../../artifacts/run1/model.pkl ./artifacts/run1/model.pkl

# --- Fix permissions ---
RUN chown -R appuser:appuser /app

# --- Switch to non-root user ---
USER appuser

# --- Expose port ---
EXPOSE 8000

# --- Start service ---
CMD ["serve-model"]
