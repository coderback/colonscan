# backend/Django.Dockerfile
FROM python:3.13-alpine

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Collect static files and apply migrations if needed (you can run this using docker-compose commands)
RUN python manage.py collectstatic --noinput

# Expose port and run gunicorn server
EXPOSE 8000
CMD ["gunicorn", "colonscan.wsgi:application", "--bind", "0.0.0.0:8000"]
