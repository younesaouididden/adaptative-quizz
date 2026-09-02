"""Benchmark A3 : adaptatif (KST) vs aleatoire vs CAT-IRT (baseline), sur la
banque piste B (domains/piste_b.yaml). Cf. ADDENDUM_BANQUE_QUESTIONS.md.

Usage : python benchmark_a3.py
Ecrit dans results/benchmark_a3/ (Lot 0, plan_action_code.md) :
  - raw.csv       : une ligne par (politique, etat simule), graine incluse
  - summary.csv   : agrege par politique
  - figure.pdf    : figure vectorielle (rapport LaTeX -- pas de PNG)
  - run.log       : commit git, date, commande, graines utilisees
Ecrit aussi benchmark_a3.png/.csv a la racine (copie de commodite pour la
demo/le rapport informel, cf. RAPPORT_AVANCEMENT.md qui l'embarque) --
results/benchmark_a3/ reste la source canonique et tracable.

Les trois politiques sont comparees sur EXACTEMENT la meme banque d'items
(35 questions, 7 concepts) et le meme processus generatif (BLIM, slip/guess)
-- seul l'algorithme de selection/diagnostic differe (meme logique que
l'amorce piste B de l'addendum : "items identiques, deux algorithmes",
personne ne peut objecter que l'ecart vient de la banque plutot que de la
methode).

Le CAT-IRT est un modele DELIBEREMENT mal specifie par rapport a la verite
terrain BLIM : theta continu unidimensionnel plaque sur un etat de
connaissance discret multi-concept. C'est precisement ce que ce benchmark
doit montrer -- pas une comparaison "loyale" entre deux modeles egalement
vrais, mais l'ecart de performance d'un vrai CAT-IRT applique a des donnees
qui suivent en realite une KST.

Limites assumees (a citer dans le rapport, cf. irt_baseline.py) : items IRT
derives de la banque BLIM (discrimination a=1.0 fixe, b depuis la difficulte
declarative 1/2/3, pas d'une calibration IRT independante) ; diagnostic par
concept de l'IRT lu via un seuil = b moyen des items du concept (mastery
testing standard, pas une propriete native du 3PL unidimensionnel).

Sur les parametres slip/guess de piste B eux-memes (cf. note_calibration.md) :
guess=0.25 est une borne combinatoire conservatrice (guess<=1/k pour un QCM a
k options), pas une valeur choisie ; slip=0.10 est un point de reference sur
un axe a balayer (aucun argument de premier principe ne fixe slip), pas une
"valeur experte". Ce benchmark mesure donc le GAIN ALGORITHMIQUE des
politiques de selection a un point de cet axe -- la formule fermee
kst_engine.questions_needed() predit 19.2 questions pour la politique
adaptative sur ce domaine, la mesure ci-dessous en donne 18.6 (accord a 3%,
cf. note_calibration.md §5). Reste a verifier (Lot 3 du plan) que le
classement des politiques ne depend pas du point choisi sur l'axe slip.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domains.loader import load_domain_yaml
from irt_baseline import simulate_irt
from kst_engine import Domain, questions_needed, simulate

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "benchmark_a3"

POLICY_LABELS = {
    "adaptatif": "Adaptatif\n(gain d'info, KST)",
    "aleatoire": "Aléatoire",
    "cat_irt": "CAT-IRT\n(baseline 3PL)",
}
POLICY_COLORS = {"adaptatif": "#2E86AB", "aleatoire": "#C0C0C0", "cat_irt": "#C0504D"}


def concept_accuracy(z_hat: frozenset, z_true: frozenset, all_concepts: list[str]) -> float:
    """Fraction des concepts correctement classes maitrise/non-maitrise."""
    return sum((c in z_hat) == (c in z_true) for c in all_concepts) / len(all_concepts)


def run_benchmark(domain: Domain, meta: list[dict]) -> list[dict]:
    """Rejoue chaque etat z in Z avec les trois politiques (etudiants simules,
    verite terrain BLIM identique pour les trois -- seul l'algorithme differe,
    graine = index de l'etat dans domain.Z, donc deterministe et reproductible).

    Retourne une ligne par (politique, etat) -- c'est le CSV brut du Lot 0,
    pas seulement l'agrege."""
    concepts = [c.name for c in domain.concepts]
    records = []

    for seed, z in enumerate(domain.Z):
        results = {
            "adaptatif": simulate(domain, z, adaptive=True, seed=seed),
            "aleatoire": simulate(domain, z, adaptive=False, seed=seed),
            "cat_irt": simulate_irt(domain, meta, z, seed=seed),
        }
        for policy, r in results.items():
            records.append({
                "policy": policy,
                "seed": seed,
                "z_true": "|".join(sorted(z)) or "(vide)",
                "n_questions": r["n_questions"],
                "correct_diagnosis": r["correct_diagnosis"],
                "concept_accuracy": concept_accuracy(r["z_hat"], z, concepts),
            })
    return records


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for policy in POLICY_LABELS:
        rows = [r for r in records if r["policy"] == policy]
        n_q = np.array([r["n_questions"] for r in rows], dtype=float)
        exact = np.array([r["correct_diagnosis"] for r in rows], dtype=float)
        concept_acc = np.array([r["concept_accuracy"] for r in rows], dtype=float)
        summary[policy] = {
            "n_questions_mean": float(n_q.mean()),
            "n_questions_std": float(n_q.std()),
            "exact_match_rate": float(exact.mean()),
            "concept_accuracy_mean": float(concept_acc.mean()),
        }
    return summary


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "seed", "z_true", "n_questions",
                                          "correct_diagnosis", "concept_accuracy"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(summary: dict[str, dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["politique", "n_questions_moyen", "n_questions_std",
                   "exactitude_etat_exact", "exactitude_par_concept"])
        for policy, s in summary.items():
            w.writerow([policy, f"{s['n_questions_mean']:.2f}", f"{s['n_questions_std']:.2f}",
                       f"{s['exact_match_rate']:.3f}", f"{s['concept_accuracy_mean']:.3f}"])


def fig_summary(summary: dict[str, dict[str, float]], path: Path) -> None:
    policies = list(POLICY_LABELS.keys())
    n_q = [summary[p]["n_questions_mean"] for p in policies]
    acc = [summary[p]["concept_accuracy_mean"] * 100 for p in policies]
    colors = [POLICY_COLORS[p] for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    b1 = ax1.bar([POLICY_LABELS[p] for p in policies], n_q, color=colors)
    ax1.set_ylabel("Nombre moyen de questions")
    ax1.set_title("Questions pour atteindre l'arrêt")
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center")

    b2 = ax2.bar([POLICY_LABELS[p] for p in policies], acc, color=colors)
    ax2.set_ylabel("Exactitude par concept (%)")
    ax2.set_title("Précision du diagnostic final")
    ax2.set_ylim(0, 100)
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Benchmark A3 — domaine piste B (7 concepts, |Z|=50), "
                 "moyenne sur tous les états simulés")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "inconnu (git indisponible)"


def write_run_log(path: Path, n_states: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)     : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande       : python benchmark_a3.py\n")
        f.write(f"commit git     : {_git_commit()}\n")
        f.write(f"domaine        : {DOMAIN_PATH.relative_to(ROOT)}\n")
        f.write(f"graines        : seed = index de l'etat dans domain.Z, "
               f"0..{n_states - 1} (deterministe, {n_states} etats)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)
    records = run_benchmark(domain, meta)
    summary = summarize(records)

    predicted = questions_needed(slip=0.10, guess=0.25, n_concepts=len(domain.concepts))
    print(f"prediction fermee (questions_needed) : {predicted:.1f} questions "
         f"(adaptatif, cf. note_calibration.md)")
    for policy, s in summary.items():
        print(f"{policy:10s} n_questions={s['n_questions_mean']:5.2f}±{s['n_questions_std']:.2f}  "
             f"exact={s['exact_match_rate']:.1%}  concept_acc={s['concept_accuracy_mean']:.1%}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(summary, RESULTS_DIR / "summary.csv")
    fig_summary(summary, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", n_states=len(domain.Z))
    print(f"ecrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    # copies de commodite a la racine (demo / RAPPORT_AVANCEMENT.md)
    write_summary_csv(summary, ROOT / "benchmark_a3_table.csv")
    fig_summary(summary, ROOT / "benchmark_a3.png")
    print("ecrit (copies racine) : benchmark_a3_table.csv, benchmark_a3.png")


if __name__ == "__main__":
    main()
