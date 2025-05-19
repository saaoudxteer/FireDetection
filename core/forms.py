from django import forms

class UploadForm(forms.Form):
    file = forms.FileField(
        label="Choisissez une image ou une vidéo",
        widget=forms.ClearableFileInput(attrs={'accept': 'image/*,video/*'})
    )
