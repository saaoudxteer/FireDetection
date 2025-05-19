# core/model.py

# Pondérations optimisées
WEIGHTS = {
    "fire": 0.40,          # Incendie
    "uncontrolled": 0.25,  # Situation non contrôlée
    "smoke": 0.15,         # Fumée
    "person": 0.10,        # Présence humaine
    "forest": 0.10,        # Environnement (forêt)
}

# Libellés
DANGER_LABELS = ["Pas de danger", "Danger", "Danger extrême"]

def compute_danger_score(fire, smoke, person, uncontrolled, forest):
    p = 1.0 if person else 0.0
    s = (
       ( (1-fire) * WEIGHTS["fire"])
      + uncontrolled * WEIGHTS["uncontrolled"]
      + smoke * WEIGHTS["smoke"]
      + p * WEIGHTS["person"]
      + forest * WEIGHTS["forest"]
    )
    # (Somme WEIGHTS vaut déjà 1.0)
    return s

def compute_danger_level(score):
    """
    Niveau :
      - 0 Pas de danger : score < 0.25
      - 1 Danger      : 0.25 ≤ score < 0.60
      - 2 Danger extrême : score ≥ 0.60
    """
    if score < 0.25:
        return 0
    if score < 0.60:
        return 1
    return 2
