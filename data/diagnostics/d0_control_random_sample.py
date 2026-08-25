"""
Controle recommande par docs/revue_d0_tour2.md (action prioritaire, C-ctrl) :
la comparaison entre responses.parquet (25,9M lignes) et
responses_first_attempt.parquet (2,28M lignes) n'est pas a variables
controlees -- deux choses changent en meme temps, le TYPE d'observation
(premiere tentative vs n'importe laquelle) et la QUANTITE d'information
(~11x moins de lignes par paire etudiant/concept). Ce controle isole la
quantite seule : pour chaque (etudiant, concept), on tire au hasard AUTANT
d'observations que dans la variante premiere-tentative, mais parmi TOUTES
les tentatives de ce couple (pas seulement problem_number==1).

Astuce memoire-legere (0,6 Go de RAM libre sur cette machine) : le nombre
de reponses correctes dans un tirage SANS REMISE de taille K depuis un sac
de N items dont C corrects suit exactement une loi HYPERGEOMETRIQUE. Pas
besoin d'acceder aux 25,9M lignes individuelles -- les comptes agreges
(n_correct, n_total) par paire, deja produits par
calibrate_em.sufficient_statistics (teste dans test_calibrate_em.py), sont
suffisants pour simuler exactement ce que donnerait un tirage ligne par
ligne.

Lecture : si ce controle reproduit l'explosion de slip -> c'est un
artefact de la MAIGREUR des donnees (moins d'observations = moins
d'information pour separer slip de guess), pas du filtre
premiere-tentative en soi. S'il garde guess proche des valeurs de
responses.parquet -> le filtre premiere-tentative fait bien le travail
et le changement observe vient bien du TYPE d'observation.

Ecrit dans data/diagnostics/, ne touche jamais domain.yaml ni
responses.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import calibrate_em  # reutilise sufficient_statistics / run_em -- pas duplique

VARIANT_PATH = Path(__file__).resolve().parent / "responses_first_attempt.parquet"


def build_control_statistics(responses_path: Path = calibrate_em.RESPONSES_PATH,
                             variant_path: Path = VARIANT_PATH,
                             domain_path: Path = calibrate_em.DOMAIN_PATH,
                             seed: int = 0):
    """Reduit les deux jeux a des statistiques suffisantes, puis tire
    n_correct_control ~ Hypergeometrique(bon=n_correct_complet,
    mauvais=n_total_complet-n_correct_complet, echantillon=K) ou K est le
    n_total de la variante pour cette paire (etudiant, concept).
    Mathematiquement exact, pas une approximation du tirage ligne-par-ligne.
    """
    concepts, Z = calibrate_em.load_domain(domain_path)

    full = pd.read_parquet(responses_path)
    n_correct_full, n_total_full, students_full = calibrate_em.sufficient_statistics(full, concepts)
    del full

    variant = pd.read_parquet(variant_path)
    n_correct_var, K, students_var = calibrate_em.sufficient_statistics(variant, concepts)
    del variant

    # les etudiants de la variante sont un sous-ensemble de ceux du jeu
    # complet (memes filtres de concept) -- alignement explicite, jamais
    # suppose sur l'ordre des deux pivots.
    student_idx_full = {s: i for i, s in enumerate(students_full)}
    idx_map = np.array([student_idx_full[s] for s in students_var])

    n_correct_pool = n_correct_full[idx_map]
    n_total_pool = n_total_full[idx_map]

    if (K > n_total_pool + 1e-9).any():
        raise ValueError("La variante contient plus d'observations que le pool complet "
                        "pour au moins une paire (etudiant, concept) -- verifier "
                        "l'alignement ou la coherence des deux extractions.")

    rng = np.random.default_rng(seed)
    n_correct_control = np.zeros_like(K)
    mask = K > 0
    n_correct_control[mask] = rng.hypergeometric(
        ngood=n_correct_pool[mask].astype(np.int64),
        nbad=(n_total_pool[mask] - n_correct_pool[mask]).astype(np.int64),
        nsample=K[mask].astype(np.int64)).astype(float)

    return n_correct_control, K, concepts, Z


if __name__ == "__main__":
    n_correct_control, n_total_control, concepts, Z = build_control_statistics()

    rng_master = np.random.default_rng(1)
    results = []
    for _ in range(5):
        rng = np.random.default_rng(rng_master.integers(2**32 - 1))
        results.append(calibrate_em.run_em(n_correct_control, n_total_control, Z, concepts,
                                           rng, max_iter=100, tol=1e-4))
    best = max(results, key=lambda r: r.log_likelihood_trace[-1])

    print(f"Convergence : {best.converged}")
    finals = np.array([r.log_likelihood_trace[-1] for r in results])
    slips = np.array([r.slip for r in results])
    guesses = np.array([r.guess for r in results])
    print(f"log-vraisemblance finale (5 restarts) : {finals}")
    print()
    print(f"{'concept':22s} {'slip':>7s} {'(ecart-type)':>13s} {'guess':>7s} {'(ecart-type)':>13s}")
    for i, c in enumerate(concepts):
        print(f"  {c:20s} {best.slip[i]:7.3f} {slips[:, i].std():13.4f} "
              f"{best.guess[i]:7.3f} {guesses[:, i].std():13.4f}")
