from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.base import ContentFile
import base64
import requests
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
@permission_classes([AllowAny])
def health_check(request):
    return Response({'status': 'ColonoScan backend is up and running'})


class SlideViewSet(viewsets.ModelViewSet):
    queryset = Slide.objects.none()
    serializer_class = SlideSerializer
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return Slide.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        # 1) create the job first
        job = AnalysisJob.objects.create()
        
        # 2) pass job into serializer.save() so that
        #    Slide.objects.create(..., job=job) never inserts NULL
        slide = serializer.save(owner=self.request.user, job=job)
        
        # 3) kick off Celery task
        from .tasks import analyze_slide
        analyze_slide.delay(slide.id)


class PatchViewSet(viewsets.ModelViewSet):
    queryset = Patch.objects.none()
    serializer_class = PatchSerializer
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

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

    @action(detail=False, methods=['post'], url_path='batch')
    def batch_analysis(self, request):
        """
        Analyze multiple patches at once.
        Expects multiple files in the request.
        """
        files = request.FILES.getlist('files')
        if not files:
            return Response({'error': 'No files provided'}, status=400)

        results = []
        
        for file in files:
            try:
                # Create a job for this patch
                job = AnalysisJob.objects.create()
                
                # Create the patch record
                patch = Patch.objects.create(
                    owner=request.user,
                    image=file,
                    job=job
                )

                # Call the histopathology service directly for immediate results
                try:
                    with open(patch.image.path, "rb") as f:
                        files_data = [("files", (file.name, f, f"image/{file.name.split('.')[-1]}"))]
                        params = {"patch_size": 224}
                        resp = requests.post("http://histopathology:8001/infer/patch", files=files_data, params=params)
                    
                    if resp.status_code == 200:
                        data = resp.json()[0]  # Get first result since we're processing one file at a time
                        
                        # Update patch with results
                        patch.predicted_class = data["predicted_class"]
                        patch.class_name = data["class_name"]
                        patch.probabilities = data["probabilities"]

                        # Save GradCAM and saliency images
                        if "gradcam" in data:
                            gradcam_bytes = base64.b64decode(data["gradcam"])
                            patch.gradcam_file.save(
                                f"{patch.id}_gradcam.png",
                                ContentFile(gradcam_bytes),
                                save=False
                            )

                        if "saliency" in data:
                            saliency_bytes = base64.b64decode(data["saliency"])
                            patch.saliency_file.save(
                                f"{patch.id}_saliency.png",
                                ContentFile(saliency_bytes),
                                save=False
                            )

                        patch.save()
                        job.status = AnalysisJob.COMPLETED
                        job.save()

                        # Prepare result for frontend
                        result = {
                            'id': patch.id,
                            'image_url': request.build_absolute_uri(patch.image.url),
                            'class_name': patch.class_name,
                            'probabilities': patch.probabilities,
                            'gradcam_url': request.build_absolute_uri(patch.gradcam_file.url) if patch.gradcam_file else None,
                            'saliency_url': request.build_absolute_uri(patch.saliency_file.url) if patch.saliency_file else None,
                        }
                        results.append(result)
                    else:
                        job.status = AnalysisJob.FAILED
                        job.log = f"Service returned status {resp.status_code}"
                        job.save()
                        results.append({
                            'error': f'Analysis failed for {file.name}',
                            'filename': file.name
                        })
                        
                except Exception as e:
                    job.status = AnalysisJob.FAILED
                    job.log = str(e)
                    job.save()
                    results.append({
                        'error': f'Analysis failed for {file.name}: {str(e)}',
                        'filename': file.name
                    })
                    
            except Exception as e:
                results.append({
                    'error': f'Failed to process {file.name}: {str(e)}',
                    'filename': file.name
                })

        return Response(results)


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
