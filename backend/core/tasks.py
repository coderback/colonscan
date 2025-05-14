import base64, tempfile
import requests
from io import BytesIO
from django.core.files.base import ContentFile
from django.utils import timezone
from celery import shared_task
from .models import Slide, Patch, VideoSession, GenomicSample, AnalysisJob

SLIDE_URL = 'http://histopathology:8001/infer/slide'
PATCH_URL = "http://histopathology:8001/infer/patch"
VIDEO_URL = 'http://colonoscopy:8002/detect'
GENOMIC_URL = 'http://genomic:8003/run'


@shared_task
def analyze_slide(slide_id):
    slide = Slide.objects.get(pk=slide_id)
    job = slide.job
    job.status = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        # stream-slide file
        with open(slide.slide_file.path, "rb") as f:
            files = {"file": (slide.slide_file.name, f, "application/octet-stream")}
            params = {"patch_size": 224, "overlap": 0.5}
            resp = requests.post(SLIDE_URL, files=files, params=params)
        resp.raise_for_status()
        data = resp.json()

        # summary text from your FastAPI
        slide.summary = data["summary"]

        # optional: decode an overview heatmap if you returned one
        if "overview_map" in data:
            b64 = data["overview_map"]
            img_bytes = base64.b64decode(b64)
            slide.overview_map.save(
                f"{slide_id}_overview.png",
                ContentFile(img_bytes),
                save=False
            )

        slide.save()
        job.status = AnalysisJob.COMPLETED

    except Exception as exc:
        job.status = AnalysisJob.FAILED
        job.log = str(exc)

    finally:
        job.finished_at = timezone.now()
        job.save()


@shared_task
def analyze_patch(patch_id):
    patch = Patch.objects.get(pk=patch_id)
    job = patch.job
    job.status = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        with open(patch.image.path, "rb") as f:
            files = [("files", (patch.image.name, f,  f"image/{patch.image.name.split('.')[-1]}"))]
            params = {"patch_size": 224}
            resp = requests.post(PATCH_URL, files=files, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Populate patch record
        patch.predicted_class = data["predicted_class"]
        patch.class_name = data["class_name"]
        patch.probabilities = data["probabilities"]

        # decode & save gradcam
        gradcam_b64 = data["gradcam"]
        gradcam_bytes = base64.b64decode(gradcam_b64)
        patch.gradcam_file.save(
            f"{patch_id}_gradcam.png",
            ContentFile(gradcam_bytes),
            save=False
        )

        # decode & save saliency
        sal_b64 = data["saliency"]
        sal_bytes = base64.b64decode(sal_b64)
        patch.saliency_file.save(
            f"{patch_id}_saliency.png",
            ContentFile(sal_bytes),
            save=False
        )

        patch.save()
        job.status = AnalysisJob.COMPLETED

    except Exception as exc:
        job.status = AnalysisJob.FAILED
        job.log = str(exc)

    finally:
        job.finished_at = timezone.now()
        job.save()


@shared_task
def analyze_genomic(sample_id):
    sample = GenomicSample.objects.get(pk=sample_id)
    job = sample.job
    job.status = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        # call genomic microservice
        resp = requests.post(GENOMIC_URL, json={'path': sample.sample_file.path})
        resp.raise_for_status()
        result = resp.json()
        vcf_path = result['vcf_path']
        metrics = result['metrics']

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
        job.log = str(e)

    finally:
        job.finished_at = timezone.now()
        job.save()
