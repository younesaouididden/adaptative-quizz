"""Lot 3.3 -- graphe de prerequis faux (plan_action_code.md).

15% des etudiants simules ont un etat vrai z_true QUI N'EST PAS dans
domain.Z (viole la structure de prerequis de piste B -- ex.
algebra_advanced maitrise sans algebra_linear). Le moteur ne peut
structurellement PAS representer un tel etat : z_hat = domain.Z[argmax(p)]
est toujours un etat VALIDE, donc correct_diagnosis (z_hat==z_true) est
FAUX par construction pour ces etudiants -- ce n'est pas la bonne question.
La bonne metrique est la distance de Hamming entre z_hat et z_true (nombre
de concepts ou le diagnostic differe de la verite), qui reste bien definie
meme quand z_true est structurellement irrepresentable.

Usage : python lot3_3_prereq_faux.py
Ecrit results/lot3_3_prereq_faux/{raw.csv, summary.csv, figure.pdf, run.log}
+ une copie lot3_3_prereq_faux_figure.png a la racine.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from itertools import chain, combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domains.loader import load_domain_yaml
from kst_engine import Domain, simulate

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot3_3_prereq_faux"

N_STUDENTS = 400
FRACTION_INVALID = 0.15
MASTER_SEED = 0


def hamming_distance(z1: frozenset, z2: frozenset, all_concepts: list[str]) -> int:
    """Nombre de concepts ou le diagnostic differe de la verite -- bien
    definie meme si z1 ou z2 n'est pas un etat valide de domain.Z."""
    return sum(1 for c in all_concepts if (c in z1) != (c in z2))


def invalid_states(domain: Domain) -> list[frozenset]:
    """Tous les sous-ensembles de concepts qui violent la structure de
    prerequis de piste B (absents de domain.Z) -- le powerset complet
    (2^7=128) moins les 50 etats valides."""
    names = [c.name for c in domain.concepts]
    valid = set(domain.Z)
    powerset = chain.from_iterable(combinations(names, r) for r in range(len(names) + 1))
    return [frozenset(s) for s in powerset if frozenset(s) not in valid]


def run(domain: Domain) -> list[dict]:
    concepts = [c.name for c in domain.concepts]
    invalid_pool = invalid_states(domain)
    print(f"etats invalides possibles : {len(invalid_pool)} / {2 ** len(concepts)} "
         f"sous-ensembles ({len(domain.Z)} valides)")

    rng_group = np.random.default_rng(MASTER_SEED)
    n_invalid = int(round(N_STUDENTS * FRACTION_INVALID))
    n_valid = N_STUDENTS - n_invalid

    valid_idx = rng_group.integers(0, len(domain.Z), size=n_valid)
    invalid_idx = rng_group.integers(0, len(invalid_pool), size=n_invalid)

    records = []
    for i, idx in enumerate(valid_idx):
        z_true = domain.Z[idx]
        r = simulate(domain, z_true, adaptive=True, seed=1_000_000 + i)
        records.append({
            "group": "valide (z_true dans Z)", "z_true": "|".join(sorted(z_true)) or "(vide)",
            "n_questions": r["n_questions"], "correct_diagnosis": r["correct_diagnosis"],
            "hamming": hamming_distance(r["z_hat"], z_true, concepts),
        })
    for i, idx in enumerate(invalid_idx):
        z_true = invalid_pool[idx]
        r = simulate(domain, z_true, adaptive=True, seed=2_000_000 + i)
        records.append({
            "group": "invalide (z_true hors Z)", "z_true": "|".join(sorted(z_true)) or "(vide)",
            "n_questions": r["n_questions"], "correct_diagnosis": r["correct_diagnosis"],
            "hamming": hamming_distance(r["z_hat"], z_true, concepts),
        })
    return records


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "z_true", "n_questions",
                                          "correct_diagnosis", "hamming"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["groupe", "n", "n_questions_moyen", "exactitude_etat_exact",
                   "hamming_moyen", "hamming_std"])
        for group in sorted({r["group"] for r in records}):
            rows = [r for r in records if r["group"] == group]
            n_q = np.array([r["n_questions"] for r in rows], dtype=float)
            exact = np.array([r["correct_diagnosis"] for r in rows], dtype=float)
            ham = np.array([r["hamming"] for r in rows], dtype=float)
            w.writerow([group, len(rows), f"{n_q.mean():.2f}", f"{exact.mean():.3f}",
                       f"{ham.mean():.3f}", f"{ham.std():.3f}"])


def fig_summary(records: list[dict], n_concepts: int, path: Path) -> None:
    groups = sorted({r["group"] for r in records})
    colors = {groups[0]: "#2E86AB", groups[1]: "#C0504D"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for group in groups:
        ham = [r["hamming"] for r in records if r["group"] == group]
        bins = np.arange(-0.5, n_concepts + 1.5, 1)
        ax1.hist(ham, bins=bins, alpha=0.6, color=colors[group], label=group, density=True)
    ax1.set_xlabel("Distance de Hamming (concepts mal diagnostiqués)")
    ax1.set_ylabel("Densité")
    ax1.set_title("Distribution de la distance de Hamming")
    ax1.legend(fontsize=8)

    means = [np.mean([r["n_questions"] for r in records if r["group"] == g]) for g in groups]
    b = ax2.bar(groups, means, color=[colors[g] for g in groups])
    ax2.set_ylabel("Nombre moyen de questions")
    ax2.set_title("Coût en questions")
    ax2.tick_params(axis="x", labelrotation=10)
    for bar in b:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Lot 3.3 — {FRACTION_INVALID:.0%} d'états vrais hors Z "
                f"(prérequis violés), piste B")
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


def write_run_log(path: Path, n_invalid_pool: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)        : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande          : python lot3_3_prereq_faux.py\n")
        f.write(f"commit git        : {_git_commit()}\n")
        f.write(f"domaine           : {DOMAIN_PATH.relative_to(ROOT)}\n")
        f.write(f"n_etudiants       : {N_STUDENTS} ({FRACTION_INVALID:.0%} hors Z)\n")
        f.write(f"etats invalides possibles : {n_invalid_pool}\n")
        f.write(f"graines           : MASTER_SEED={MASTER_SEED} pour le tirage des etats ; "
               f"seed=1000000+i (valides) / 2000000+i (invalides) pour simulate()\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)

    records = run(domain)
    groups = sorted({r["group"] for r in records})
    for group in groups:
        rows = [r for r in records if r["group"] == group]
        n_q = np.mean([r["n_questions"] for r in rows])
        exact = np.mean([r["correct_diagnosis"] for r in rows])
        ham = np.mean([r["hamming"] for r in rows])
        print(f"{group:28s} n={len(rows):3d}  n_questions={n_q:.2f}  "
             f"exact={exact:.1%}  hamming_moyen={ham:.3f}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(records, RESULTS_DIR / "summary.csv")
    fig_summary(records, len(domain.concepts), RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", len(invalid_states(domain)))
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    fig_summary(records, len(domain.concepts), ROOT / "lot3_3_prereq_faux_figure.png")
    print("ecrit (copie racine) : lot3_3_prereq_faux_figure.png")


if __name__ == "__main__":
    main()
