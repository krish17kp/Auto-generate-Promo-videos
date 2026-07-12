import uuid

from django.conf import settings
from django.db import models


class VideoUpload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.CASCADE)
    file = models.FileField(upload_to="input/")
    original_name = models.CharField(max_length=255)
    size_bytes = models.BigIntegerField(default=0)
    duration_s = models.FloatField(default=0)
    fps = models.FloatField(default=0)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    has_audio = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)


class PromoJob(models.Model):
    STATUS_CHOICES = [
        ("queued", "queued"),
        ("processing", "processing"),
        ("done", "done"),
        ("failed", "failed"),
    ]
    STAGE_CHOICES = [
        ("ingest", "ingest"),
        ("scenes", "scenes"),
        ("features", "features"),
        ("scoring", "scoring"),
        ("narrative", "narrative"),
        ("render", "render"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    upload = models.OneToOneField(VideoUpload, on_delete=models.CASCADE, related_name="job")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="queued", db_index=True)
    stage = models.CharField(max_length=16, choices=STAGE_CHOICES, null=True, blank=True)
    progress = models.SmallIntegerField(default=0)
    params = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        indexes = [models.Index(fields=["status", "created_at"])]


class Scene(models.Model):
    job = models.ForeignKey(PromoJob, on_delete=models.CASCADE, related_name="scenes", db_index=True)
    index = models.SmallIntegerField()
    start_s = models.FloatField()
    end_s = models.FloatField()
    embedding = models.JSONField(null=True, blank=True)

    class Meta:
        unique_together = [("job", "index")]


class SegmentScore(models.Model):
    ROLE_CHOICES = [("hook", "hook"), ("build", "build"), ("climax", "climax"), ("outro", "outro")]

    scene = models.OneToOneField(Scene, on_delete=models.CASCADE, related_name="score")
    visual = models.FloatField(null=True, blank=True)
    audio = models.FloatField(null=True, blank=True)
    motion = models.FloatField(null=True, blank=True)
    quality = models.FloatField(null=True, blank=True)
    transcript = models.FloatField(null=True, blank=True)
    fused = models.FloatField(db_index=True)
    selected = models.BooleanField(default=False)
    narrative_role = models.CharField(max_length=8, choices=ROLE_CHOICES, null=True, blank=True)


class PromoOutput(models.Model):
    ASPECT_CHOICES = [("16:9", "16:9"), ("9:16", "9:16"), ("1:1", "1:1")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(PromoJob, on_delete=models.CASCADE, related_name="outputs")
    file = models.FileField(upload_to="output/")
    aspect = models.CharField(max_length=8, choices=ASPECT_CHOICES, default="16:9")
    duration_s = models.FloatField(default=0)
    size_bytes = models.BigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)


class EvalRun(models.Model):
    job = models.ForeignKey(PromoJob, null=True, blank=True, on_delete=models.SET_NULL, related_name="eval_runs")
    config = models.JSONField(default=dict, blank=True)
    precision_at_5 = models.FloatField()
    auprc = models.FloatField()
    ci_low = models.FloatField()
    ci_high = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
