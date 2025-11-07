from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from django.shortcuts import render

def home(request):
    return render(request, 'index.html')

def generate_video(request):
    if request.method == "POST" and request.FILES.get('video'):
        video_file = request.FILES['video']

        # Save input video
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'input'))
        input_path = fs.save(video_file.name, video_file)
        input_url = fs.url(input_path)

        # --- Mock promo video generation ---
        # In your real code, call your CNN/video generator function here.
        # For now, we simulate by copying the same video as output.
        output_folder = os.path.join(settings.MEDIA_ROOT, 'output')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "promo_" + video_file.name)
        with open(os.path.join(settings.MEDIA_ROOT, 'input', video_file.name), 'rb') as inp, open(output_path, 'wb') as out:
            out.write(inp.read())

        output_url = settings.MEDIA_URL + "output/promo_" + video_file.name

        return render(request, 'result.html', {'video_url': output_url})
    return render(request, 'index.html')
