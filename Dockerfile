FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# collectstatic only needs Django to boot, not a real secret — the deploy platform's
# runtime env var overrides this at container start (Dockerfile ENV is a default, not a lock-in).
ENV SECRET_KEY=build-time-placeholder-override-at-runtime
RUN python manage.py collectstatic --noinput

ENV DEBUG=False
EXPOSE 8000

CMD ["gunicorn", "promo_project.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "120"]
