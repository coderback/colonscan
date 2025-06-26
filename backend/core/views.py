from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.base import ContentFile
import base64
import requests
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.authtoken.models import Token
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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analytics_dashboard(request):
    """
    Get comprehensive analytics for the dashboard.
    """
    user = request.user
    
    # Get date ranges
    now = timezone.now()
    last_30_days = now - timedelta(days=30)
    last_7_days = now - timedelta(days=7)
    
    # Slides analytics
    slides = Slide.objects.filter(owner=user)
    slides_stats = {
        'total': slides.count(),
        'completed': slides.filter(job__status='COMPLETED').count(),
        'pending': slides.filter(job__status='PENDING').count(),
        'failed': slides.filter(job__status='FAILED').count(),
        'last_30_days': slides.filter(created__gte=last_30_days).count(),
        'last_7_days': slides.filter(created__gte=last_7_days).count(),
    }
    
    # Videos analytics
    videos = VideoSession.objects.filter(owner=user)
    videos_stats = {
        'total': videos.count(),
        'completed': videos.filter(job__status='COMPLETED').count(),
        'pending': videos.filter(job__status='PENDING').count(),
        'failed': videos.filter(job__status='FAILED').count(),
        'last_30_days': videos.filter(uploaded__gte=last_30_days).count(),
        'last_7_days': videos.filter(uploaded__gte=last_7_days).count(),
    }
    
    # Genomic analytics
    genomic = GenomicSample.objects.filter(owner=user)
    genomic_stats = {
        'total': genomic.count(),
        'completed': genomic.filter(job__status='COMPLETED').count(),
        'pending': genomic.filter(job__status='PENDING').count(),
        'failed': genomic.filter(job__status='FAILED').count(),
        'last_30_days': genomic.filter(uploaded__gte=last_30_days).count(),
        'last_7_days': genomic.filter(uploaded__gte=last_7_days).count(),
    }
    
    # Overall performance metrics
    all_jobs = AnalysisJob.objects.filter(
        Q(slide__owner=user) | Q(videosession__owner=user) | Q(genomicsample__owner=user)
    )
    
    total_jobs = all_jobs.count()
    completed_jobs = all_jobs.filter(status='COMPLETED').count()
    success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    
    # Recent activity (last 10 items)
    recent_slides = slides.order_by('-created')[:3]
    recent_videos = videos.order_by('-uploaded')[:3]
    recent_genomic = genomic.order_by('-uploaded')[:4]
    
    recent_activity = []
    
    for slide in recent_slides:
        recent_activity.append({
            'type': 'slide',
            'id': slide.id,
            'title': slide.slide_file.name.split('/')[-1],
            'status': slide.job.status if slide.job else 'PENDING',
            'timestamp': slide.created,
            'summary': slide.summary
        })
    
    for video in recent_videos:
        recent_activity.append({
            'type': 'video',
            'id': video.id,
            'title': video.video_file.name.split('/')[-1],
            'status': video.job.status if video.job else 'PENDING',
            'timestamp': video.uploaded,
            'summary': f"Video analysis - {video.resolution or 'Unknown resolution'}"
        })
    
    for sample in recent_genomic:
        recent_activity.append({
            'type': 'genomic',
            'id': sample.id,
            'title': sample.sample_file.name.split('/')[-1],
            'status': sample.job.status if sample.job else 'PENDING',
            'timestamp': sample.uploaded,
            'summary': f"Genomic analysis - {sample.sample_type}"
        })
    
    # Sort by timestamp and take top 10
    recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
    recent_activity = recent_activity[:10]
    
    # Model performance stats (mock data for now)
    model_stats = {
        'wsi_accuracy': 94.2,
        'polyp_detection': 89.7,
        'genomic_accuracy': 91.3,
        'overall_accuracy': 91.7
    }
    
    # Processing time trends (mock data)
    processing_trends = [
        {'date': '2024-01', 'avg_time': 3.2},
        {'date': '2024-02', 'avg_time': 2.8},
        {'date': '2024-03', 'avg_time': 2.5},
        {'date': '2024-04', 'avg_time': 2.1},
        {'date': '2024-05', 'avg_time': 1.9},
        {'date': '2024-06', 'avg_time': 1.7}
    ]
    
    return Response({
        'slides': slides_stats,
        'videos': videos_stats,
        'genomic': genomic_stats,
        'performance': {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'success_rate': round(success_rate, 1),
            'avg_processing_time': 2.5  # minutes
        },
        'recent_activity': recent_activity,
        'model_stats': model_stats,
        'processing_trends': processing_trends,
        'growth': {
            'slides_30d': slides_stats['last_30_days'],
            'videos_30d': videos_stats['last_30_days'],
            'genomic_30d': genomic_stats['last_30_days'],
            'total_30d': slides_stats['last_30_days'] + videos_stats['last_30_days'] + genomic_stats['last_30_days']
        }
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    """
    Register a new user account.
    """
    try:
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')
        confirm_password = request.data.get('confirm_password')
        first_name = request.data.get('first_name', '')
        last_name = request.data.get('last_name', '')

        # Validation
        if not all([username, email, password, confirm_password]):
            return Response({
                'error': 'All fields are required'
            }, status=status.HTTP_400_BAD_REQUEST)

        if password != confirm_password:
            return Response({
                'error': 'Passwords do not match'
            }, status=status.HTTP_400_BAD_REQUEST)

        if len(password) < 8:
            return Response({
                'error': 'Password must be at least 8 characters long'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            return Response({
                'error': 'Username already exists'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return Response({
                'error': 'Email already registered'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )

        # Create auth token
        token, created = Token.objects.get_or_create(user=user)

        return Response({
            'message': 'User registered successfully',
            'token': token.key,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name
            }
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response({
            'error': 'Registration failed. Please try again.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
