from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from .models import Slide, Patch, VideoSession, GenomicSample, AnalysisJob
from .serializers import (
    SlideSerializer,
    PatchSerializer,
    VideoSessionSerializer,
    GenomicSampleSerializer,
    AnalysisJobSerializer
)


# Create your views here.

@api_view(['GET'])
def health_check(request):
    return Response({'status': 'ColonoScan backend is up and running'})


class SlideViewSet(viewsets.ModelViewSet):
    queryset = Slide.objects.none()
    serializer_class = SlideSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Slide.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        slide = serializer.save(owner=self.request.user)
        job = AnalysisJob.objects.create()
        slide.job = job
        slide.save()
        # kick off Celery task
        from .tasks import analyze_slide
        analyze_slide.delay(slide.id)


class PatchViewSet(viewsets.ModelViewSet):
    queryset = Patch.objects.none()
    serializer_class = PatchSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Patch.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        # 1) create the job
        job = AnalysisJob.objects.create()

        # 2) pass job into serializer.save() so that
        #    Patch.objects.create(..., job=job) never inserts NULL
        patch = serializer.save(owner=self.request.user, job=job)

        # 3) kick off your async task
        from .tasks import analyze_patch
        analyze_patch.delay(patch.id)


class VideoSessionViewSet(viewsets.ModelViewSet):
    queryset = VideoSession.objects.none()
    serializer_class = VideoSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return VideoSession.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        video = serializer.save(owner=self.request.user)
        job = AnalysisJob.objects.create()
        video.job = job
        video.save()
        from .tasks import analyze_video
        analyze_video.delay(video.id)


class GenomicSampleViewSet(viewsets.ModelViewSet):
    queryset = GenomicSample.objects.none()
    serializer_class = GenomicSampleSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return GenomicSample.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        sample = serializer.save(owner=self.request.user)
        job = AnalysisJob.objects.create()
        sample.job = job
        sample.save()
        from .tasks import analyze_genomic
        analyze_genomic.delay(sample.id)


class AnalysisJobViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AnalysisJob.objects.all()
    serializer_class = AnalysisJobSerializer
    permission_classes = [IsAuthenticated]
