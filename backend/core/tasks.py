import base64
import requests
from io import BytesIO
from django.core.files.base import ContentFile
from django.utils import timezone
from celery import shared_task
from .models import Slide, VideoSession, GenomicSample, AnalysisJob

SLIDE_URL   = 'http://histopathology:8001/infer'
VIDEO_URL   = 'http://colonoscopy:8002/detect'
GENOMIC_URL = 'http://genomic:8003/run'

@shared_task
def analyze_slide(slide_id):
    slide = Slide.objects.get(pk=slide_id)
    job   = slide.job
    job.status     = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        resp = requests.post(SLIDE_URL, json={'path': slide.slide_file.path})
        resp.raise_for_status()
        data = resp.json()

        # classification
        slide.classification = data['classification']

        # helper to decode and save each map
        def save_map(key, field, prefix):
            img_bytes = base64.b64decode(data[key])
            buf = ContentFile(img_bytes)
            setattr(
                slide,
                field,
                buf
            )
            # Django needs the name on save()
            getattr(slide, field).save(f"{prefix}_{slide_id}.png", buf, save=False)

        save_map('saliency', 'saliency_file', 'saliency')
        save_map('gradcam',  'gradcam_file',  'gradcam')
        save_map('shap',     'shap_file',     'shap')

        slide.save()
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