#!/usr/bin/env python
import os
import sys
import django

# Add the backend directory to the Python path
sys.path.append('./backend')

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'colonscan.settings')
django.setup()

from core.models import AnalysisJob, Slide

print("=== Analysis Jobs Status ===")
print(f"Total jobs: {AnalysisJob.objects.count()}")
print(f"Failed jobs: {AnalysisJob.objects.filter(status='FAILED').count()}")
print(f"Completed jobs: {AnalysisJob.objects.filter(status='COMPLETED').count()}")
print(f"Running jobs: {AnalysisJob.objects.filter(status='RUNNING').count()}")

print("\n=== Recent Failed Jobs ===")
failed_jobs = AnalysisJob.objects.filter(status='FAILED').order_by('-created_at')[:5]
for job in failed_jobs:
    print(f"Job {job.id}: {job.created_at}")
    print(f"  Log: {job.log[:200] if job.log else 'No log'}")
    print(f"  Started: {job.started_at}")
    print(f"  Finished: {job.finished_at}")
    print()

print("=== Recent Completed Jobs ===")
completed_jobs = AnalysisJob.objects.filter(status='COMPLETED').order_by('-created_at')[:3]
for job in completed_jobs:
    print(f"Job {job.id}: {job.created_at}")
    print(f"  Started: {job.started_at}")
    print(f"  Finished: {job.finished_at}")
    print() 