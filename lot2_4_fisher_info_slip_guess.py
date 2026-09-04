"""Lot 2.4 -- deplacement geometrique attendu (Fisher-Rao) en fonction de
(slip, guess) (plan_action_code.md).

Courbe de kst_engine.expected_fisher_rao_step(slip=0.10, guess) pour guess
dans [0.01, 0.89], avec les valeurs REELLES piste A (calibrees par EM,
data/domain.yaml) et piste B (bornee, domains/piste_b.yaml) marquees dessus
-- chaque point utilise le VRAI (slip, guess) du concept, pas seulement sa
projection sur la courbe a slip fixe.

Effet narratif (plan) : quand guess monte, les probabilites d'emission
P(y|a,z) pour des etats z differents se rapprochent, l'item cesse de
separer les etats, son deplacement geometrique attendu s'effondre. La saga
du guess degenere (piste A, PROMPT_A2_GRANULARITE.md) devient une
PREDICTION QUANTITATIVE de la theorie verifiee sur donnees reelles, pas
seulement une limite documentee apres coup.

Usage : python lot2_4_fisher_info_slip_guess.py
Ecrit results/lot2_4_fisher_info/{raw.csv, figure.pdf, run.log} + une copie
lot2_4_fisher_info_figure.png a la racine.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from kst_engine import expected_fisher_rao_step, item_information

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "lot2_4_fisher_info"
PISTE_A_DOMAIN = ROOT / "data" / "domain.yaml"
PISTE_B_DOMAIN = ROOT / "domains" / "piste_b.yaml"
BASELINE_SLIP = 0.10


def load_concepts(path: Path) -> list[tuple[str, float, float]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [(c["id"], c["slip"], c["guess"]) for c in data["concepts"]]


def fig_collapse(guess_grid: np.ndarray, geo_curve: list[float],
                 piste_a: list[tuple[str, float, float]],
                 piste_b: list[tuple[str, float, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(guess_grid, geo_curve, color="black", linewidth=2,
           label=f"courbe à slip={BASELINE_SLIP} fixe")

    for concept_id, slip, guess in piste_a:
        val = expected_fisher_rao_step(slip=slip, guess=guess)
        ax.scatter(guess, val, color="#C0504D", s=70, zorder=5)
        ax.annotate(concept_id, (guess, val), textcoords="offset points",
                   xytext=(6, 4), fontsize=8, color="#C0504D")
    ax.scatter([], [], color="#C0504D", s=70, label="piste A (calibrée EM, par concept)")

    concept_id, slip, guess = piste_b[0]  # tous les concepts piste B partagent (0.10, 0.25)
    val = expected_fisher_rao_step(slip=slip, guess=guess)
    ax.scatter(guess, val, color="#2E86AB", s=100, marker="D", zorder=6,
              label=f"piste B (guess={guess}, tous concepts)")

    ax.set_xlabel("guess")
    ax.set_ylabel("Déplacement géométrique attendu par question (distance de Fisher-Rao)")
    ax.set_title("Lot 2.4 — effondrement du déplacement géométrique quand guess augmente\n"
                "(piste A : guess calibré par EM, 0,50–0,75 ; piste B : guess borné à 0,25)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_raw_csv(piste_a: list[tuple[str, float, float]],
                  piste_b: list[tuple[str, float, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "concept", "slip", "guess",
                   "expected_fisher_rao_step", "item_information_nats"])
        for source, rows in (("piste_a", piste_a), ("piste_b", piste_b)):
            for concept_id, slip, guess in rows:
                w.writerow([source, concept_id, slip, guess,
                          f"{expected_fisher_rao_step(slip=slip, guess=guess):.6f}",
                          f"{item_information(slip=slip, guess=guess):.6f}"])


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "inconnu (git indisponible)"


def write_run_log(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC) : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande   : python lot2_4_fisher_info_slip_guess.py\n")
        f.write(f"commit git : {_git_commit()}\n")
        f.write(f"sources    : {PISTE_A_DOMAIN.relative_to(ROOT)}, "
               f"{PISTE_B_DOMAIN.relative_to(ROOT)}\n")
        f.write(f"aucune graine aleatoire (calcul deterministe, pas de simulation)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    guess_grid = np.linspace(0.01, 1.0 - BASELINE_SLIP - 0.01, 200)
    geo_curve = [expected_fisher_rao_step(slip=BASELINE_SLIP, guess=float(g))
                for g in guess_grid]

    piste_a = load_concepts(PISTE_A_DOMAIN)
    piste_b = load_concepts(PISTE_B_DOMAIN)

    print("piste A (calibrée EM) :")
    for concept_id, slip, guess in piste_a:
        print(f"  {concept_id:25s} slip={slip:.4f} guess={guess:.4f}  "
             f"deplacement={expected_fisher_rao_step(slip=slip, guess=guess):.4f}  "
             f"item_information={item_information(slip=slip, guess=guess):.4f} nats")
    print("piste B (bornée) :")
    concept_id, slip, guess = piste_b[0]
    print(f"  (tous les concepts)     slip={slip:.4f} guess={guess:.4f}  "
         f"deplacement={expected_fisher_rao_step(slip=slip, guess=guess):.4f}  "
         f"item_information={item_information(slip=slip, guess=guess):.4f} nats")

    fig_collapse(guess_grid, geo_curve, piste_a, piste_b, RESULTS_DIR / "figure.pdf")
    write_raw_csv(piste_a, piste_b, RESULTS_DIR / "raw.csv")
    write_run_log(RESULTS_DIR / "run.log")
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, figure.pdf, run.log}}")

    fig_collapse(guess_grid, geo_curve, piste_a, piste_b, ROOT / "lot2_4_fisher_info_figure.png")
    print("ecrit (copie racine) : lot2_4_fisher_info_figure.png")


if __name__ == "__main__":
    main()
