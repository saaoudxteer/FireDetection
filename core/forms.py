from django import forms

class UploadForm(forms.Form):
    file = forms.FileField(
        label="Choisissez une image ou une vidéo",
        widget=forms.ClearableFileInput(attrs={'accept': 'image/*,video/*'})
    )
class DetectionForm(forms.Form):
    image = forms.ImageField(
        label="Image à analyser",
        help_text="Formats acceptés : JPEG, PNG, etc.",
        widget=forms.ClearableFileInput(attrs={
            "accept": "image/*"
        })
    )