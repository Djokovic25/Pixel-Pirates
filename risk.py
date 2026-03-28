def compute_risk(label, size):
    morphology_scores = {
        "Fiber": 90,
        "Fragment": 70,
        "Film": 60,
        "Pellet": 40
    }

    morph = morphology_scores.get(label, 50)
    size_score = max(0, 100 - size * 0.2)

    return int(min(0.6 * morph + 0.4 * size_score, 100))