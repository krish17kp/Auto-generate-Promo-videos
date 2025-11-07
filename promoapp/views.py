from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import UploadForm
import subprocess, os
from pathlib import Path

def index(request):
    form = UploadForm()
    return render(request, 'promoapp/index.html', {'form': form})

def generate_promo(request):
    if request.method == 'POST' and request.FILES.get('video'):
        f = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(f.name, f)
        input_path = fs.path(filename)
        output_path = os.path.join(settings.MEDIA_ROOT, f'promo_{f.name}.mp4')

        # Run your generator script
        cmd = f'python promoapp/promo4.1.py --input "{input_path}" --output "{output_path}" --duration 30 --fps 2 --save-scores'
        subprocess.run(cmd, shell=True, check=False)

        return render(request, 'promoapp/index.html', {
            'form': UploadForm(),
            'output_video': f'/media/promo_{f.name}.mp4'
        })
    return render(request, 'promoapp/index.html', {'form': UploadForm()})
