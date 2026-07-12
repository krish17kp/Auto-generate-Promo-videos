from django import forms

ASPECT_CHOICES = [("16:9", "16:9 — landscape"), ("9:16", "9:16 — vertical"), ("1:1", "1:1 — square")]
DURATION_CHOICES = [(15, "15s"), (30, "30s"), (60, "60s")]
PROFILE_CHOICES = [
    ("mvp", "MVP — fast (audio + frame-diff)"),
    ("capstone", "Capstone — hybrid (CLIP + motion + quality)"),
    ("advanced", "Advanced — + transcript & captions"),
]


class UploadForm(forms.Form):
    video = forms.FileField()
    aspect = forms.ChoiceField(choices=ASPECT_CHOICES, initial="16:9", required=False)
    all_formats = forms.BooleanField(required=False, help_text="Render 16:9, 9:16, and 1:1 in one job")
    duration = forms.TypedChoiceField(choices=DURATION_CHOICES, coerce=int, initial=30, required=False)
    profile = forms.ChoiceField(choices=PROFILE_CHOICES, initial="mvp", required=False)
    title = forms.CharField(max_length=100, required=False)
    cta = forms.CharField(max_length=100, required=False)
