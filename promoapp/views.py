from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from .forms import UploadForm
import subprocess
import os
import time

# ------------------------------
# Homepage (upload form)
# ------------------------------
def index(request):
    form = UploadForm()
    return render(request, 'promoapp/index.html', {'form': form})

# ------------------------------
# Original video generation route (heavy)
# ------------------------------
def generate_promo(request):
    if request.method == 'POST' and request.FILES.get('video'):
        f = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(f.name, f)
        input_path = fs.path(filename)
        output_path = os.path.join(settings.MEDIA_ROOT, f'promo_{f.name}.mp4')

        # ⚠️ This is the heavy process that can crash Railway free tier
        cmd = f'python promoapp/promo4.1.py --input "{input_path}" --output "{output_path}" --duration 30 --fps 2 --save-scores'
        subprocess.run(cmd, shell=True, check=False)

        return render(request, 'promoapp/index.html', {
            'form': UploadForm(),
            'output_video': f'/media/promo_{f.name}.mp4'
        })
    return render(request, 'promoapp/index.html', {'form': UploadForm()})

# ------------------------------
# Lightweight test route for Railway (/generate/)
# ------------------------------
def generate(request):
    """
    Temporary lightweight endpoint for Railway testing.
    Simulates promo video generation to confirm app runs fine online.
    """
    time.sleep(3)  # Simulate short processing delay
    return HttpResponse("✅ Promo video generation simulated successfully!")
