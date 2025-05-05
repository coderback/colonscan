"""
URL configuration for colonscan project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from core.views import health_check
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken import views as drf_views
from core.views import (
    SlideViewSet,
    VideoSessionViewSet,
    GenomicSampleViewSet,
    AnalysisJobViewSet,
    health_check
)

router = DefaultRouter()
router.register(r'slides', SlideViewSet, basename='slide')
router.register(r'videos', VideoSessionViewSet, basename='video')
router.register(r'genomic', GenomicSampleViewSet, basename='genomic')
router.register(r'jobs', AnalysisJobViewSet, basename='job')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/health/', health_check),
    path('api/', include(router.urls)),
    path('api/auth/login/', drf_views.obtain_auth_token, name='api_token_auth'),
]