from pathlib import Path

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django_ratelimit.decorators import ratelimit

from .forms import UploadForm
from .jobs import start_job
from .models import PromoJob, Scene, VideoUpload
from .pipeline import ingest
from .pipeline.run import PROFILE_OVERRIDES


def home(request):
    return render(request, "promoapp/index.html")


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
    else:
        form = UserCreationForm()
    return render(request, "promoapp/signup.html", {"form": form})


@login_required
def history(request):
    jobs = (
        PromoJob.objects.filter(upload__owner=request.user)
        .select_related("upload")
        .order_by("-created_at")
    )
    return render(request, "promoapp/history.html", {"jobs": jobs})


@ratelimit(key="ip", rate="10/m", block=True)
def generate_video(request):
    if request.method != "POST":
        return render(request, "promoapp/index.html")

    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid() or not request.FILES.get("video"):
        return render(request, "promoapp/index.html", {"error": "please choose a video file"})

    video_file = request.FILES["video"]
    ext = Path(video_file.name).suffix.lower()
    if ext not in ingest.ALLOWED_EXTENSIONS:
        return render(request, "promoapp/index.html", {"error": f"unsupported file type: {ext or '(none)'}"})

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if video_file.size > max_bytes:
        error = f"file too large: {video_file.size / 1e6:.0f}MB (max {settings.MAX_UPLOAD_MB:.0f}MB)"
        return render(request, "promoapp/index.html", {"error": error})

    fs = FileSystemStorage(location=str(Path(settings.MEDIA_ROOT) / "input"))
    saved_name = fs.save(video_file.name, video_file)
    saved_path = Path(fs.location) / saved_name

    try:
        info = ingest.probe(saved_path)
        ingest.validate(info, saved_path.stat().st_size, settings.MAX_UPLOAD_MB, settings.MAX_UPLOAD_MINUTES)
    except ingest.IngestError as exc:
        saved_path.unlink(missing_ok=True)
        return render(request, "promoapp/index.html", {"error": str(exc)})

    owner = request.user if request.user.is_authenticated else None
    upload = VideoUpload.objects.create(
        owner=owner,
        file=f"input/{saved_name}",
        original_name=video_file.name,
        size_bytes=saved_path.stat().st_size,
        duration_s=info.duration,
        fps=info.fps,
        width=info.width,
        height=info.height,
        has_audio=info.has_audio,
    )

    cleaned = form.cleaned_data
    aspect = cleaned.get("aspect") or "16:9"
    duration = cleaned.get("duration") or 30
    profile = cleaned.get("profile") or "mvp"
    aspects = ["16:9", "9:16", "1:1"] if cleaned.get("all_formats") else [aspect]

    job = PromoJob.objects.create(
        upload=upload,
        params={
            "aspect": aspect,
            "aspects": aspects,
            "profile": profile,
            "config": {
                "target_duration": duration,
                "aspect": aspect,
                "title": cleaned.get("title") or None,
                "cta": cleaned.get("cta") or None,
                **PROFILE_OVERRIDES.get(profile, {}),
            },
        },
    )
    start_job(job.id)

    return redirect("job_result", job_id=job.id)


def job_status(request, job_id):
    job = get_object_or_404(PromoJob, id=job_id)
    return JsonResponse(
        {"status": job.status, "stage": job.stage, "progress": job.progress, "error": job.error_message}
    )


def job_result(request, job_id):
    job = get_object_or_404(PromoJob.objects.select_related("upload"), id=job_id)

    if job.status in ("queued", "processing"):
        return render(request, "promoapp/progress.html", {"job": job})
    if job.status == "failed":
        return render(request, "promoapp/index.html", {"error": job.error_message})

    outputs = list(job.outputs.order_by("aspect"))
    scenes = Scene.objects.filter(job=job).select_related("score").order_by("index")
    return render(
        request,
        "promoapp/result.html",
        {
            "job": job,
            "outputs": outputs,
            "video_url": outputs[0].file.url if outputs else None,
            "scenes": scenes,
        },
    )
