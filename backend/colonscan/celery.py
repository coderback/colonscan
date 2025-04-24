import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'colonscan.settings')

app = Celery('colonscan')
# read broker / backend URLs from Django settings prefixed with CELERY_
app.config_from_object('django.conf:settings', namespace='CELERY')
# auto-discover @shared_task in your INSTALLED_APPS
app.autodiscover_tasks()
