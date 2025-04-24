from rest_framework import serializers
from .models import Slide, VideoSession, GenomicSample, AnalysisJob

class AnalysisJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = ['id', 'status', 'created_at', 'started_at', 'finished_at', 'log']

class SlideSerializer(serializers.ModelSerializer):
    status     = serializers.CharField(source='job.status', read_only=True)
    result_url = serializers.SerializerMethodField()

    class Meta:
        model = Slide
        fields = ['id', 'slide_file', 'stain', 'uploaded', 'status', 'result_url']

    def get_result_url(self, obj):
        request = self.context.get('request')
        if obj.result_file:
            return request.build_absolute_uri(obj.result_file.url)
        return None

class VideoSessionSerializer(serializers.ModelSerializer):
    status     = serializers.CharField(source='job.status', read_only=True)
    result     = serializers.JSONField(source='result_data', read_only=True)

    class Meta:
        model = VideoSession
        fields = ['id', 'video_file', 'frame_rate', 'resolution', 'uploaded', 'status', 'result']

class GenomicSampleSerializer(serializers.ModelSerializer):
    status = serializers.CharField(source='job.status', read_only=True)
    vcf    = serializers.SerializerMethodField()
    metrics= serializers.JSONField(read_only=True)

    class Meta:
        model = GenomicSample
        fields = ['id', 'sample_file', 'sample_type', 'uploaded', 'status', 'vcf', 'metrics']

    def get_vcf(self, obj):
        request = self.context.get('request')
        if obj.vcf_file:
            return request.build_absolute_uri(obj.vcf_file.url)
        return None