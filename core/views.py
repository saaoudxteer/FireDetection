import os
from datetime import datetime

from django.conf import settings
from django.shortcuts import render, redirect

from .forms import UploadForm


def home(request):
    return render(request, 'home.html', {
        'year': datetime.now().year
    })


def upload(request):
    """
    Affiche le formulaire et, en POST, sauvegarde le fichier,
    lance le modèle (simulé ici) et renvoie le résultat.
    """
    result = None
    form = UploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        f = form.cleaned_data['file']
        # Crée le dossier media s’il n’existe pas
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        save_path = settings.MEDIA_ROOT / f.name

        # Sauvegarde du fichier
        with open(save_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        # TODO : remplacer cette simulation par ton véritable appel au modèle
        result = {
            'is_fire': True,                  # True si feu détecté
            'confidence': 92.3,               # % de confiance
            'file_type': f.name.split('.')[-1],
            'processing_time': 0.95,          # en secondes
        }

    return render(request, 'features.html', {
        'form': form,
        'result': result,
        'year': datetime.now().year
    })


def result(request, filename):
    """
    Vue complémentaire si tu souhaites conserver une page séparée
    de ‘result’. Sinon, tu peux l’enlever et tout gérer dans upload().
    """
    return render(request, 'result.html', {
        'filename': filename,
        'file_url': settings.MEDIA_URL + filename,
        'year': datetime.now().year
    })


