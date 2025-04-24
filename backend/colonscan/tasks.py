import base64
import requests
from io import BytesIO
from django.core.files.base import ContentFile
from django.utils import timezone
from celery import shared_task
from .models import Slide, VideoSession, GenomicSample, AnalysisJob

SLIDE_URL   = 'http://slide-service:8001/infer'
VIDEO_URL   = 'http://video-service:8002/detect'
GENOMIC_URL = 'http://genomic-service:8003/run'

@shared_task
def analyze_slide(slide_id):
    slide = Slide.objects.get(pk=slide_id)
    job   = slide.job
    job.status     = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        # call slide microservice
        resp = requests.post(SLIDE_URL, json={'path': slide.slide_file.path})
        resp.raise_for_status()
        data = resp.json()
        # heatmap is base64-encoded
        png_bytes = base64.b64decode(data['heatmap'])

        # save result_file
        slide.result_file.save(
            f"heatmap_{slide_id}.png",
            ContentFile(png_bytes),
            save=True
        )

        job.status = AnalysisJob.COMPLETED

    except Exception as e:
        job.status = AnalysisJob.FAILED
        job.log    = str(e)

    finally:
        job.finished_at = timezone.now()
        job.save()


@shared_task
def analyze_genomic(sample_id):
    sample = GenomicSample.objects.get(pk=sample_id)
    job    = sample.job
    job.status     = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        # call genomic microservice
        resp = requests.post(GENOMIC_URL, json={'path': sample.sample_file.path})
        resp.raise_for_status()
        result = resp.json()
        vcf_path = result['vcf_path']
        metrics  = result['metrics']

        # read & save VCF
        with open(vcf_path, 'rb') as f:
            sample.vcf_file.save(f"sample_{sample_id}.vcf",
                                 ContentFile(f.read()),
                                 save=True)

        sample.metrics = metrics
        sample.save()

        job.status = AnalysisJob.COMPLETED

    except Exception as e:
        job.status = AnalysisJob.FAILED
        job.log    = str(e)

    finally:
        job.finished_at = timezone.now()
        job.save()