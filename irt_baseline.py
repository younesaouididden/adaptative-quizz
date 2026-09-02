"""Baseline CAT-IRT (3PL, critere d'information de Fisher) pour le benchmark A3.

Sert de point de comparaison a la selection KST (kst_engine.pi_star) :
theta continu unidimensionnel au lieu d'un etat de connaissance discret sur
plusieurs concepts. Modele DELIBEREMENT mal specifie par rapport a la verite
terrain BLIM -- c'est le point du benchmark (cf. docstring de benchmark_a3.py).

Parametres d'item derives de la banque BLIM existante, pas d'une calibration
IRT independante (limite assumee, a citer dans le rapport) :
  - c (pseudo-guessing) = question.guess
  - b (difficulte)      = (difficulte declarative YAML - 2), donc {-1, 0, 1}
  - a (discrimination)  = 1.0 fixe pour tous les items

Estimation de theta par EAP (esperance a posteriori) sur une grille, plutot
que par MLE : la MLE diverge vers +/-infini tant que les reponses sont
uniformement correctes/incorrectes (frequent en debut de CAT), l'EAP avec
prior N(0,1) reste bornee des la premiere reponse -- c'est la pratique
standard des systemes CAT reels.
"""

from __future__ import annotations

import numpy as np

from kst_engine import Domain

THETA_GRID = np.linspace(-4.0, 4.0, 81)
_PRIOR = np.exp(-0.5 * THETA_GRID ** 2)
_PRIOR /= _PRIOR.sum()


def p_correct_3pl(theta: np.ndarray | float, a: float, b: float, c: float):
    """P(correct | theta) sous le modele logistique a 3 parametres."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def fisher_info_3pl(theta: float, a: float, b: float, c: float) -> float:
    """I(theta) pour un item 3PL (Lord 1980) : (P')^2 / (P(1-P))."""
    p = p_correct_3pl(theta, a, b, c)
    return a ** 2 * ((p - c) / (1.0 - c)) ** 2 * (1.0 - p) / p


def eap_theta(responses: list[tuple[float, float, float, bool]],
             grid: np.ndarray = THETA_GRID) -> tuple[float, float]:
    """Theta EAP et son erreur standard, a partir de (a, b, c, correct) deja observes."""
    w = _PRIOR.copy()
    for a, b, c, correct in responses:
        p = p_correct_3pl(grid, a, b, c)
        w = w * (p if correct else (1.0 - p))
        w = w / w.sum()
    theta_hat = float(np.sum(grid * w))
    se = float(np.sqrt(np.sum(w * (grid - theta_hat) ** 2)))
    return theta_hat, se


def build_irt_items(domain: Domain, meta: list[dict],
                    discrimination: float = 1.0) -> list[dict]:
    """Deduit (a, b, c) par question, aligne sur domain.questions."""
    items = []
    for q, m in zip(domain.questions, meta):
        difficulty = m.get("difficulty") or 2
        items.append({"a": discrimination, "b": float(difficulty - 2), "c": q.guess})
    return items


def select_next_irt(theta_hat: float, items: list[dict],
                    asked: set[int]) -> tuple[int | None, float]:
    """Argmax de l'information de Fisher parmi les items non poses."""
    best, best_info = None, -np.inf
    for i, it in enumerate(items):
        if i in asked:
            continue
        info = fisher_info_3pl(theta_hat, it["a"], it["b"], it["c"])
        if info > best_info:
            best, best_info = i, info
    return best, best_info


def simulate_irt(domain: Domain, meta: list[dict], z_true: frozenset,
                 seed: int = 0, max_questions: int = 30,
                 se_stop: float = 0.3) -> dict:
    """Fait passer le quiz a un etudiant simule (verite terrain BLIM) selon
    la politique CAT-IRT : selection par info de Fisher, arret sur SE(theta),
    diagnostic par concept via seuil = b moyen des items du concept."""
    rng = np.random.default_rng(seed)
    items = build_irt_items(domain, meta)
    responses: list[tuple[float, float, float, bool]] = []
    asked: set[int] = set()

    theta_hat, se = eap_theta(responses)
    trace = [se]

    while True:
        if len(asked) >= min(max_questions, len(items)) or se <= se_stop:
            break
        q, _ = select_next_irt(theta_hat, items, asked)
        if q is None:
            break
        question = domain.questions[q]
        # l'etudiant repond selon le vrai BLIM, pas selon le 3PL (le 3PL est
        # le modele DE L'ALGORITHME, pas la verite terrain -- cf. docstring)
        true_p = (1 - question.slip) if question.concept in z_true else question.guess
        correct = bool(rng.random() < true_p)

        it = items[q]
        responses.append((it["a"], it["b"], it["c"], correct))
        asked.add(q)
        theta_hat, se = eap_theta(responses)
        trace.append(se)

    concept_bs: dict[str, list[float]] = {}
    for q, it in zip(domain.questions, items):
        concept_bs.setdefault(q.concept, []).append(it["b"])
    z_hat = frozenset(c for c, bs in concept_bs.items() if theta_hat >= np.mean(bs))

    return {"n_questions": len(asked), "se_trace": trace, "theta_hat": theta_hat,
            "se": se, "z_hat": z_hat, "correct_diagnosis": z_hat == z_true}
