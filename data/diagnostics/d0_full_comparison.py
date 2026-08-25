"""
Comparaison consolidee des 3 runs (docs/revue_d0_tour2.md, C3+C4) :
- complet (responses.parquet, 25.9M lignes)
- variante premiere-tentative (responses_first_attempt.parquet, 2.28M lignes)
- controle (meme quantite d'information que la variante, mais tiree au
  hasard parmi TOUTES les tentatives -- cf. d0_control_random_sample.py)

Imprime, pour les 3 runs : slip/guess avec ecart-type sur 5 restarts (C4),
et la marginale P(concept maitrise) sous le prior calibre (C3) -- jamais
imprimee jusqu'ici alors que le sujet du diagnostic EST un parametre
degenere, la premiere chose a inspecter est la distribution sur Z.

Ne touche jamais domain.yaml ni responses.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import calibrate_em

from d0_control_random_sample import build_control_statistics

VARIANT_PATH = Path(__file__).resolve().parent / "responses_first_attempt.parquet"


def concept_marginals_from_prior(prior: np.ndarray, Z: list[frozenset],
                                 concepts: list[str]) -> dict[str, float]:
    """P(concept maitrise) = somme de prior(z) sur les z contenant le
    concept -- chapitre 3, meme formule que kst_engine.concept_marginals,
    reecrite ici car calibrate_em ne construit pas d'objet Domain."""
    return {c: float(sum(prior[k] for k, z in enumerate(Z) if c in z))
           for c in concepts}


def run_5_restarts(n_correct, n_total, Z, concepts, seed=0, max_iter=100, tol=1e-4):
    rng_master = np.random.default_rng(seed)
    results = []
    for _ in range(5):
        rng = np.random.default_rng(rng_master.integers(2**32 - 1))
        results.append(calibrate_em.run_em(n_correct, n_total, Z, concepts, rng,
                                           max_iter=max_iter, tol=tol))
    best = max(results, key=lambda r: r.log_likelihood_trace[-1])
    return best, results


def report(label: str, best, results, concepts, Z):
    slips = np.array([r.slip for r in results])
    guesses = np.array([r.guess for r in results])
    marg = concept_marginals_from_prior(best.prior, Z, concepts)
    print(f"\n=== {label} (convergence={best.converged}) ===")
    print(f"{'concept':22s} {'slip':>7s} {'(sd)':>8s} {'guess':>7s} {'(sd)':>8s} {'P(maitrise)':>12s}")
    for i, c in enumerate(concepts):
        print(f"  {c:20s} {best.slip[i]:7.3f} {slips[:, i].std():8.4f} "
              f"{best.guess[i]:7.3f} {guesses[:, i].std():8.4f} {marg[c]:12.3f}")


if __name__ == "__main__":
    concepts, Z = calibrate_em.load_domain()

    print("--- Run complet (responses.parquet) ---")
    import pandas as pd
    full = pd.read_parquet(calibrate_em.RESPONSES_PATH)
    n_correct_full, n_total_full, _ = calibrate_em.sufficient_statistics(full, concepts)
    del full
    best_full, res_full = run_5_restarts(n_correct_full, n_total_full, Z, concepts, seed=0)
    report("COMPLET (25.9M lignes)", best_full, res_full, concepts, Z)

    print("\n--- Run variante premiere-tentative ---")
    variant = pd.read_parquet(VARIANT_PATH)
    n_correct_var, n_total_var, _ = calibrate_em.sufficient_statistics(variant, concepts)
    del variant
    best_var, res_var = run_5_restarts(n_correct_var, n_total_var, Z, concepts, seed=1)
    report("VARIANTE 1ERE TENTATIVE (2.28M lignes)", best_var, res_var, concepts, Z)

    print("\n--- Run controle (meme quantite d'info, tirage au hasard) ---")
    n_correct_ctrl, n_total_ctrl, _, _ = build_control_statistics(seed=0)
    best_ctrl, res_ctrl = run_5_restarts(n_correct_ctrl, n_total_ctrl, Z, concepts, seed=2)
    report("CONTROLE (meme n qu'ci-dessus, tirage aleatoire)", best_ctrl, res_ctrl, concepts, Z)
