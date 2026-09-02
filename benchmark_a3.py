"""Benchmark A3 : adaptatif (KST) vs aleatoire vs CAT-IRT (baseline), sur la
banque piste B (domains/piste_b.yaml). Cf. ADDENDUM_BANQUE_QUESTIONS.md.

Usage : python benchmark_a3.py
Ecrit benchmark_a3_table.csv et benchmark_a3.png a la racine du repo.

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
testing standard, pas une propriete native du 3PL unidimensionnel). Comme le
domaine piste B n'est pas calibre empiriquement, ce benchmark mesure le GAIN
ALGORITHMIQUE des politiques de selection sur une banque donnee, pas une
validation empirique des parametres eux-memes (celle-ci reste du ressort de
la piste A, sur donnees reelles Junyi Academy).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domains.loader import load_domain_yaml
from irt_baseline import simulate_irt
from kst_engine import Domain, simulate

DOMAIN_PATH = Path(__file__).resolve().parent / "domains" / "piste_b.yaml"
OUT_DIR = Path(__file__).resolve().parent

POLICY_LABELS = {
    "adaptatif": "Adaptatif\n(gain d'info, KST)",
    "aleatoire": "Aléatoire",
    "cat_irt": "CAT-IRT\n(baseline 3PL)",
}
POLICY_COLORS = {"adaptatif": "#2E86AB", "aleatoire": "#C0C0C0", "cat_irt": "#C0504D"}


def concept_accuracy(z_hat: frozenset, z_true: frozenset, all_concepts: list[str]) -> float:
    """Fraction des concepts correctement classes maitrise/non-maitrise."""
    return sum((c in z_hat) == (c in z_true) for c in all_concepts) / len(all_concepts)


def run_benchmark(domain: Domain, meta: list[dict]) -> dict[str, list[tuple]]:
    """Rejoue chaque etat z in Z avec les trois politiques (etudiants simules,
    verite terrain BLIM identique pour les trois -- seul l'algorithme differe)."""
    concepts = [c.name for c in domain.concepts]
    rows: dict[str, list[tuple]] = {"adaptatif": [], "aleatoire": [], "cat_irt": []}

    for seed, z in enumerate(domain.Z):
        r_adapt = simulate(domain, z, adaptive=True, seed=seed)
        r_rand = simulate(domain, z, adaptive=False, seed=seed)
        r_irt = simulate_irt(domain, meta, z, seed=seed)

        for policy, r in (("adaptatif", r_adapt), ("aleatoire", r_rand), ("cat_irt", r_irt)):
            rows[policy].append((
                r["n_questions"],
                r["correct_diagnosis"],
                concept_accuracy(r["z_hat"], z, concepts),
            ))
    return rows


def summarize(rows: dict[str, list[tuple]]) -> dict[str, dict[str, float]]:
    summary = {}
    for policy, vals in rows.items():
        n_q = np.array([v[0] for v in vals], dtype=float)
        exact = np.array([v[1] for v in vals], dtype=float)
        concept_acc = np.array([v[2] for v in vals], dtype=float)
        summary[policy] = {
            "n_questions_mean": float(n_q.mean()),
            "n_questions_std": float(n_q.std()),
            "exact_match_rate": float(exact.mean()),
            "concept_accuracy_mean": float(concept_acc.mean()),
        }
    return summary


def write_table(summary: dict[str, dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["politique", "n_questions_moyen", "n_questions_std",
                   "exactitude_etat_exact", "exactitude_par_concept"])
        for policy, s in summary.items():
            w.writerow([policy, f"{s['n_questions_mean']:.2f}", f"{s['n_questions_std']:.2f}",
                       f"{s['exact_match_rate']:.3f}", f"{s['concept_accuracy_mean']:.3f}"])


def fig_summary(summary: dict[str, dict[str, float]], path: Path) -> None:
    policies = list(summary.keys())
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
    print(f"ecrit : {path}")


def main() -> None:
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)
    rows = run_benchmark(domain, meta)
    summary = summarize(rows)

    for policy, s in summary.items():
        print(f"{policy:10s} n_questions={s['n_questions_mean']:5.2f}±{s['n_questions_std']:.2f}  "
             f"exact={s['exact_match_rate']:.1%}  concept_acc={s['concept_accuracy_mean']:.1%}")

    table_path = OUT_DIR / "benchmark_a3_table.csv"
    write_table(summary, table_path)
    print(f"ecrit : {table_path}")
    fig_summary(summary, OUT_DIR / "benchmark_a3.png")


if __name__ == "__main__":
    main()
