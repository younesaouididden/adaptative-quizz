"""
Tache A2 (ADDENDUM_BANQUE_QUESTIONS.md) : calibration EM des parametres BLIM
(slip, guess par concept) et du prior sur Z, a partir des reponses reelles
(responses.parquet).

HYPOTHESE DE MODELISATION (limite assumee -- PERSPECTIVE FUTURE, a
documenter dans le memoire, pas a lever ici) :
Le BLIM (chapitre 3, kst_engine.py) suppose un etat de connaissance z FIXE
pendant tout le quiz. Nos logs couvrent 2012-2015 : un etudiant progresse
forcement sur une telle periode. On traite ici tout l'historique d'un
etudiant comme UNE SEULE observation de son etat -- ce n'est pas la
realite, c'est une approximation deliberee pour rester dans le perimetre
du PFA. Modeliser la progression dans le temps (z_s(t) evolutif, chaine de
Markov sur Z) est une extension naturelle mais hors perimetre : a citer
comme perspective future, pas a implementer.

Simplification cle permise par cette hypothese : sous z fixe, la
vraisemblance d'un etudiant ne depend que du NOMBRE de bonnes/mauvaises
reponses par concept, pas de leur ordre ni de leur horodatage -- on reduit
donc responses.parquet a des statistiques suffisantes (etudiant, concept,
n_correct, n_total) avant de lancer l'EM (chapitre 3 : c'est la
vraisemblance du BLIM, produit sur les reponses independantes sachant z,
qui rend cette reduction valide).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kst_engine import build_knowledge_space

DOMAIN_PATH = Path(__file__).resolve().parent / "domain.yaml"
RESPONSES_PATH = Path(__file__).resolve().parent / "responses.parquet"
EPS = 1e-9
MIN_GAP = 0.02  # slip + guess <= 1 - MIN_GAP, jamais == 1 (chapitre 3)


def load_domain(path: Path = DOMAIN_PATH) -> tuple[list[str], list[frozenset]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    concepts = [c["id"] for c in data["concepts"]]
    prereqs = [tuple(p) for p in data["prerequisites"]]
    Z = build_knowledge_space(concepts, prereqs)
    return concepts, Z


def membership_matrix(Z: list[frozenset], concepts: list[str]) -> np.ndarray:
    """M[z_idx, c_idx] = 1 si concepts[c_idx] in Z[z_idx], sinon 0."""
    return np.array([[1.0 if c in z else 0.0 for c in concepts] for z in Z])


def sufficient_statistics(responses: pd.DataFrame, concepts: list[str]
                          ) -> tuple[np.ndarray, np.ndarray, list]:
    """Reduit les reponses a (n_correct, n_total) par (etudiant, concept).

    Valide sous l'hypothese z fixe par etudiant (voir docstring module) :
    seul le COMPTE de bonnes/mauvaises reponses par concept compte pour la
    vraisemblance, jamais leur ordre.
    """
    grouped = (responses.groupby(["student_id", "concept_id"])["correct"]
              .agg(["sum", "count"]).reset_index())
    pivot_correct = grouped.pivot(index="student_id", columns="concept_id",
                                  values="sum").reindex(columns=concepts, fill_value=0)
    pivot_total = grouped.pivot(index="student_id", columns="concept_id",
                                values="count").reindex(columns=concepts, fill_value=0)
    pivot_correct = pivot_correct.fillna(0)
    pivot_total = pivot_total.fillna(0)
    return pivot_correct.to_numpy(dtype=float), pivot_total.to_numpy(dtype=float), \
        list(pivot_correct.index)


# ---------------------------------------------------------------------------
# EM : E-step et M-step (chapitre 3 : BLIM, vraisemblance des reponses
# independantes sachant z)
# ---------------------------------------------------------------------------

def e_step(n_correct: np.ndarray, n_total: np.ndarray, slip: np.ndarray,
          guess: np.ndarray, prior: np.ndarray, M: np.ndarray
          ) -> tuple[np.ndarray, float]:
    """gamma[s,z] = P(z | reponses de l'etudiant s), calcule en log-espace.

    Retourne (gamma, log-vraisemblance totale des donnees sous les
    parametres courants) -- cette derniere sert au suivi de convergence.
    """
    n_incorrect = n_total - n_correct
    P = M * (1 - slip) + (1 - M) * guess          # (|Z|, C) : P(correct | z, c)
    P = np.clip(P, EPS, 1 - EPS)
    logP, log1mP = np.log(P), np.log(1 - P)

    log_lik = n_correct @ logP.T + n_incorrect @ log1mP.T     # (n_etudiants, |Z|)
    log_joint = log_lik + np.log(prior + EPS)[None, :]
    log_norm = logsumexp(log_joint, axis=1, keepdims=True)     # (n_etudiants, 1)
    gamma = np.exp(log_joint - log_norm)
    return gamma, float(log_norm.sum())


def m_step(n_correct: np.ndarray, n_total: np.ndarray, gamma: np.ndarray,
          M: np.ndarray, slip_prev: np.ndarray, guess_prev: np.ndarray,
          min_gap: float = MIN_GAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Met a jour prior, slip, guess par MLE ponderee par gamma.

    Contrainte slip+guess<1 (chapitre 3) imposee par PROJECTION apres coup :
    si violee pour un concept, les deux sont reduits proportionnellement
    (le ratio slip/guess est preserve) jusqu'a slip+guess = 1-min_gap.
    Un concept sans masse suffisante dans un role (jamais observe "maitrise"
    ou jamais "non maitrise" sous la croyance courante) garde sa valeur
    precedente plutot qu'un NaN -- pas assez de signal pour reestimer.
    """
    prior_new = gamma.mean(axis=0)

    weight_total = gamma.T @ n_total        # (|Z|, C)
    weight_correct = gamma.T @ n_correct    # (|Z|, C)

    mastered_total = (M * weight_total).sum(axis=0)
    mastered_correct = (M * weight_correct).sum(axis=0)
    unmastered_total = ((1 - M) * weight_total).sum(axis=0)
    unmastered_correct = ((1 - M) * weight_correct).sum(axis=0)

    slip_new = np.where(mastered_total > EPS,
                        1 - np.divide(mastered_correct, mastered_total,
                                     out=np.zeros_like(mastered_total),
                                     where=mastered_total > EPS),
                        slip_prev)
    guess_new = np.where(unmastered_total > EPS,
                         np.divide(unmastered_correct, unmastered_total,
                                  out=np.zeros_like(unmastered_total),
                                  where=unmastered_total > EPS),
                         guess_prev)

    total = slip_new + guess_new
    violation = total > (1 - min_gap)
    if violation.any():
        factor = (1 - min_gap) / total[violation]
        slip_new = slip_new.copy()
        guess_new = guess_new.copy()
        slip_new[violation] *= factor
        guess_new[violation] *= factor

    return prior_new, slip_new, guess_new


@dataclass
class EMResult:
    slip: np.ndarray
    guess: np.ndarray
    prior: np.ndarray
    log_likelihood_trace: list[float] = field(default_factory=list)
    converged: bool = False


def run_em(n_correct: np.ndarray, n_total: np.ndarray, Z: list[frozenset],
          concepts: list[str], rng: np.random.Generator,
          max_iter: int = 100, tol: float = 1e-4,
          init_slip: np.ndarray | None = None,
          init_guess: np.ndarray | None = None) -> EMResult:
    M = membership_matrix(Z, concepts)
    n_concepts = len(concepts)
    slip = init_slip if init_slip is not None else rng.uniform(0.05, 0.20, n_concepts)
    guess = init_guess if init_guess is not None else rng.uniform(0.15, 0.35, n_concepts)
    prior = np.full(len(Z), 1.0 / len(Z))

    n_students = n_correct.shape[0]
    trace: list[float] = []
    converged = False
    for it in range(max_iter):
        gamma, ll = e_step(n_correct, n_total, slip, guess, prior, M)
        trace.append(ll)
        # comparaison sur la log-vraisemblance MOYENNE PAR ETUDIANT : la
        # valeur totale grandit avec n_students, donc un seuil absolu sur le
        # total serait soit trop strict (jamais atteint), soit trop laxiste
        # (arrete trop tot) selon la taille de l'echantillon.
        if it > 0 and (trace[-1] - trace[-2]) / n_students < tol:
            converged = True
            break
        prior, slip, guess = m_step(n_correct, n_total, gamma, M, slip, guess)

    return EMResult(slip=slip, guess=guess, prior=prior,
                    log_likelihood_trace=trace, converged=converged)


def calibrate(n_restarts: int = 5, seed: int = 0, max_iter: int = 100,
             tol: float = 1e-4, domain_path: Path = DOMAIN_PATH,
             responses_path: Path = RESPONSES_PATH
             ) -> tuple[EMResult, list[EMResult], list[str], list[frozenset]]:
    """Calibration complete avec verification de stabilite (plusieurs
    initialisations aleatoires independantes, chapitre 3 -- l'EM ne garantit
    qu'un optimum LOCAL de la vraisemblance)."""
    concepts, Z = load_domain(domain_path)
    responses = pd.read_parquet(responses_path)
    n_correct, n_total, _students = sufficient_statistics(responses, concepts)

    rng_master = np.random.default_rng(seed)
    results = []
    for _ in range(n_restarts):
        rng = np.random.default_rng(rng_master.integers(2**32 - 1))
        results.append(run_em(n_correct, n_total, Z, concepts, rng,
                              max_iter=max_iter, tol=tol))

    best = max(results, key=lambda r: r.log_likelihood_trace[-1])
    return best, results, concepts, Z


def write_calibrated_domain(concepts: list[str], slip: np.ndarray, guess: np.ndarray,
                            domain_path: Path = DOMAIN_PATH) -> None:
    with open(domain_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    slip_by_concept = dict(zip(concepts, slip))
    guess_by_concept = dict(zip(concepts, guess))
    for c in data["concepts"]:
        c["slip"] = round(float(slip_by_concept[c["id"]]), 4)
        c["guess"] = round(float(guess_by_concept[c["id"]]), 4)
        c["calibrated"] = True
    with open(domain_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    best, results, concepts, Z = calibrate()

    print(f"Convergence : {best.converged}   ({len(best.log_likelihood_trace)} iterations)")
    print(f"Log-vraisemblance finale (meilleur restart) : {best.log_likelihood_trace[-1]:.1f}")
    print()
    print("Stabilite sur", len(results), "initialisations independantes :")
    finals = np.array([r.log_likelihood_trace[-1] for r in results])
    print(f"  log-vraisemblance finale : {finals}")
    slips = np.array([r.slip for r in results])
    guesses = np.array([r.guess for r in results])
    print(f"  ecart-type de slip par concept  : {slips.std(axis=0).round(4)}")
    print(f"  ecart-type de guess par concept : {guesses.std(axis=0).round(4)}")
    print()
    print(f"{'concept':22s} {'slip':>7s} {'guess':>7s}")
    for c, s, g in zip(concepts, best.slip, best.guess):
        print(f"  {c:20s} {s:7.3f} {g:7.3f}")

    write_calibrated_domain(concepts, best.slip, best.guess)
    print(f"\nEcrit : {DOMAIN_PATH} (slip/guess calibres, calibrated=true)")
