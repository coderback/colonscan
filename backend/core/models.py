from django.db import models
from django.conf import settings
from django.utils import timezone


# Create your models here.
class AnalysisJob(models.Model):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (RUNNING, 'Running'),
        (COMPLETED, 'Completed'),
        (FAILED, 'Failed'),
    ]

    created_at = models.DateTimeField(default=timezone.now)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=PENDING)
    log = models.TextField(blank=True, help_text="Optional stdout/stderr")


class Slide(models.Model):
    """
    Histopathology whole-slide images.
    """
    # who owns/uploads it
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    slide_file = models.FileField(upload_to="slides/")  # your .svs or .tiff
    created = models.DateTimeField(auto_now_add=True)
    job = models.OneToOneField("AnalysisJob", on_delete=models.CASCADE)
    # results
    summary = models.TextField(blank=True)  # e.g. "Mean slide: malignant (0.8321)"
    # optional: aggregate slide‐level map (heatmap overlaid on thumbnail)
    overview_map = models.ImageField(upload_to="slide_maps/", null=True, blank=True)


# core/models.py
class Patch(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="patches/")
    created = models.DateTimeField(auto_now_add=True)
    job = models.OneToOneField("AnalysisJob", on_delete=models.CASCADE)
    # results
    predicted_class = models.IntegerField(null=True, blank=True)
    class_name = models.CharField(max_length=32, blank=True)
    probabilities = models.JSONField(null=True, blank=True)
    gradcam_file = models.ImageField(upload_to="patch_gradcams/", null=True, blank=True)
    saliency_file = models.ImageField(upload_to="patch_saliencies/", null=True, blank=True)


class VideoSession(models.Model):
    """
    Colonoscopy video for live polyp detection.
    """
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    uploaded = models.DateTimeField(auto_now_add=True)
    video_file = models.FileField(upload_to='videos/')
    frame_rate = models.FloatField(null=True, blank=True, help_text="FPS detected or declared")
    resolution = models.CharField(max_length=20, blank=True, help_text="WxH")
    job = models.OneToOneField(AnalysisJob, null=True, blank=True, on_delete=models.SET_NULL)
    result_data = models.JSONField(null=True, blank=True, help_text="Bounding boxes, timestamps")


class GenomicSample(models.Model):
    """
    Raw genomic input and downstream VCF/metrics.
    """
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    uploaded = models.DateTimeField(auto_now_add=True)
    sample_file = models.FileField(upload_to='genomic_raw/')
    sample_type = models.CharField(max_length=100, help_text="e.g. WES, RNA-seq")
    job = models.OneToOneField(AnalysisJob, null=True, blank=True, on_delete=models.SET_NULL)
    vcf_file = models.FileField(upload_to='genomic_results/', null=True, blank=True)
    metrics = models.JSONField(null=True, blank=True, help_text="Parsed variant counts, VAFs, etc.")
