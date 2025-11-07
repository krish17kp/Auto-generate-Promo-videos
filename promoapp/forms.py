from django import forms

class UploadForm(forms.Form):
    video = forms.FileField()
