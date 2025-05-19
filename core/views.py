# core/views.py

from pathlib import Path
from django.shortcuts import render
from django.conf import settings

from .forms import DetectionForm
from .model import compute_danger_score, compute_danger_level, DANGER_LABELS
from .predictors import (
    predict_fire,
    predict_smoke,
    predict_uncontrolled,
    predict_forest,
    predict_person,
)

def home(request):
    """Page d'accueil."""
    return render(request, "home.html")


def handle_upload(f):
    """Enregistre le fichier uploadé et renvoie son chemin sur le disque."""
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f.name
    with open(dest, "wb") as out:
        for chunk in f.chunks():
            out.write(chunk)
    return dest


def upload(request):
    """
    Page d'upload et d'affichage des résultats (features.html).
    Utilise DetectionForm pour n'accepter que des images.
    """
    # Initialisation du contexte avec un formulaire vide
    context = {"form": DetectionForm()}

    if request.method == "POST":
        form = DetectionForm(request.POST, request.FILES)
        if form.is_valid():
            # 1) Sauvegarde de l'image
            img_file = form.cleaned_data["image"]
            img_path = handle_upload(img_file)

            # 2) Prédictions
            fire_val         = predict_fire(img_path)
            smoke_val        = predict_smoke(img_path)
            uncontrolled_val = predict_uncontrolled(img_path)
            forest_val       = predict_forest(img_path)
            person_val       = predict_person(img_path)

            # 3) Score global et niveau
            score = compute_danger_score(
                fire_val, smoke_val, person_val, uncontrolled_val, forest_val
            )
            level = compute_danger_level(score)

            # 4) Mise à jour du contexte avec les résultats
            context.update({
                "original_url": settings.MEDIA_URL + "uploads/" + img_file.name,
                "fire_confidence": round(fire_val, 2),
                "smoke_confidence": round(smoke_val, 2),
                "uncontrolled_confidence": round(uncontrolled_val, 2),
                "environment_confidence": round(forest_val, 2),
                "person_detected": bool(person_val),
                "danger_score": round(score, 2),
                "danger_label": DANGER_LABELS[level],
            })
        else:
            # Formulaire invalide : on renvoie juste le form pour afficher les erreurs
            context = {"form": form}

    return render(request, "features.html", context)
