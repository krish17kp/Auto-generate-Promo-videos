from django.contrib import admin

from .models import EvalRun, PromoJob, PromoOutput, Scene, SegmentScore, VideoUpload

admin.site.register(VideoUpload)
admin.site.register(PromoJob)
admin.site.register(Scene)
admin.site.register(SegmentScore)
admin.site.register(PromoOutput)
admin.site.register(EvalRun)
