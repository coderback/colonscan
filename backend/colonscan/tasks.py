import base64
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
def analyze_patch(patch_id):
    patch = Patch.objects.get(pk=patch_id)
    job = patch.job
    job.status = AnalysisJob.RUNNING
    job.started_at = timezone.now()
    job.save()

    try:
        # send the patch image as multipart/form-data with field name "files"
        with open(patch.image.path, "rb") as f:
            files = [("files",
                      (patch.image.name, f, f"image/{patch.image.name.split('.')[-1]}"))]
            params = {"patch_size": 224}
            resp = requests.post(PATCH_URL, files=files, params=params)
        resp.raise_for_status()
        data = resp.json()

        # record the predictions
        patch.predicted_class = data["predicted_class"]
        patch.class_name = data["class_name"]
        patch.probabilities = data["probabilities"]

        # decode + save the Grad-CAM overlay
        gradcam_bytes = base64.b64decode(data["gradcam"])
        patch.gradcam_file.save(
            f"{patch_id}_gradcam.png",
            ContentFile(gradcam_bytes),
            save=False
        )

        # decode + save the saliency overlay
        sal_bytes = base64.b64decode(data["saliency"])
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
