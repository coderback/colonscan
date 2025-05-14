from rest_framework import serializers
from .models import Slide, Patch, VideoSession, GenomicSample, AnalysisJob


class AnalysisJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = ['id', 'status', 'created_at', 'started_at', 'finished_at', 'log']


class SlideSerializer(serializers.ModelSerializer):
    status = serializers.CharField(source="job.status", read_only=True)
    result_url = serializers.SerializerMethodField()
    summary = serializers.CharField(read_only=True)
    overview_map_url = serializers.SerializerMethodField()

    class Meta:
        model = Slide
        fields = [
            "id",
            "slide_file",
            "uploaded",
            "status",
            "summary",
            "overview_map_url",
            "result_url",
        ]

    def get_result_url(self, obj):
        """
        If you still have a per‐patch result URL field, keep this.
        Otherwise you can remove it.
        """
        request = self.context.get("request")
        if obj.result_file:
            return request.build_absolute_uri(obj.result_file.url)
        return None

    def get_overview_map_url(self, obj):
        """
        Return a fully qualified URL to the slide‐level heatmap (if generated).
        """
        request = self.context.get("request")
        if obj.overview_map:
            return request.build_absolute_uri(obj.overview_map.url)
        return None


class PatchSerializer(serializers.ModelSerializer):
    # Human‐readable class name
    class_name = serializers.CharField(source="get_predicted_class_display", read_only=True)
    # Raw probability vector
    probabilities = serializers.JSONField(read_only=True)
    # URLs to the saved overlay images
    gradcam_url = serializers.SerializerMethodField()
    saliency_url = serializers.SerializerMethodField()

    class Meta:
        model = Patch
        fields = [
            "id",
            "slide",
            "x", "y",
            "predicted_class",
            "class_name",
            "probabilities",
            "gradcam_url",
            "saliency_url",
            "created_at",
        ]

    def get_gradcam_url(self, obj):
        request = self.context.get("request")
        if obj.gradcam_map:
            return request.build_absolute_uri(obj.gradcam_map.url)
        return None

    def get_saliency_url(self, obj):
        request = self.context.get("request")
        if obj.saliency_map:
            return request.build_absolute_uri(obj.saliency_map.url)
        return None


class VideoSessionSerializer(serializers.ModelSerializer):
    status = serializers.CharField(source='job.status', read_only=True)
    result = serializers.JSONField(source='result_data', read_only=True)

    class Meta:
        model = VideoSession
        fields = ['id', 'video_file', 'frame_rate', 'resolution', 'uploaded', 'status', 'result']


class GenomicSampleSerializer(serializers.ModelSerializer):
    status = serializers.CharField(source='job.status', read_only=True)
    vcf = serializers.SerializerMethodField()
    metrics = serializers.JSONField(read_only=True)

    class Meta:
        model = GenomicSample
        fields = ['id', 'sample_file', 'sample_type', 'uploaded', 'status', 'vcf', 'metrics']

    def get_vcf(self, obj):
        request = self.context.get('request')
        if obj.vcf_file:
            return request.build_absolute_uri(obj.vcf_file.url)
        return None
