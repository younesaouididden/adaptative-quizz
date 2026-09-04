"""Lot 3.6 -- correction : un moteur prudent restaure la calibration
(plan_action_code.md).

Rejoue EXACTEMENT la meme grille de verite que 3.2 (meme graines, meme
protocole, reutilise ses fonctions), mais avec un moteur volontairement
PRUDENT : slip=0,20 / guess=0,40 supposes (au lieu des vraies valeurs
piste B, slip=0,10/guess=0,25). Un mode de defaillance ET son remede
valent mieux qu'un simple constat (cf. plan_action_code.md, 3.6).

Usage : python lot3_6_correction.py
Ecrit results/lot3_6_correction/{raw.csv, summary.csv, figure_degradation.pdf,
figure_calibration.pdf, run.log} + copies PNG a la racine.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kst_engine import Concept, Domain, Question
from lot3_2_grille_mauvaise_specification import (
    fig_calibration,
    fig_degradation,
    run_grid,
    summarize,
    write_raw_csv,
    write_run_log,
    write_summary_csv,
)

ROOT = Path(__file__).resolve().parent
PISTE_B_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot3_6_correction"

PRUDENT_SLIP, PRUDENT_GUESS = 0.20, 0.40


def load_prudent_domain(path: Path, slip: float, guess: float) -> tuple[Domain, list[dict]]:
    """Meme structure (concepts, prerequis, questions, difficultes pour
    l'IRT) que piste_b.yaml, mais slip/guess DU MOTEUR remplaces par des
    valeurs volontairement prudentes -- la verite continue d'etre fournie
    separement via verite_slip/verite_guess (simulate(), Lot 3.1)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    concepts = [Concept(c["id"]) for c in data["concepts"]]
    prereqs = [tuple(p) for p in data["prerequisites"]]
    questions = [Question(q["id"], q["concept"], slip=slip, guess=guess)
                for q in data["questions"]]
    meta = [{"stem": q["stem"], "options": q["options"], "answer": q["answer"],
            "difficulty": q.get("difficulty")} for q in data["questions"]]
    return Domain(concepts=concepts, prereqs=prereqs, questions=questions), meta


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta = load_prudent_domain(PISTE_B_PATH, PRUDENT_SLIP, PRUDENT_GUESS)

    records = run_grid(domain, meta)
    summary = summarize(records)

    print(f"moteur prudent : slip={PRUDENT_SLIP}, guess={PRUDENT_GUESS}")
    print("\nExactitude par concept (KST adaptatif), grille slip x guess :")
    from lot3_2_grille_mauvaise_specification import GUESS_GRID, SLIP_GRID
    print("slip\\guess  " + "  ".join(f"{g:.2f}" for g in GUESS_GRID))
    for vs in SLIP_GRID:
        row = [f"{summary[(vs, vg, 'kst_adaptatif')]['concept_acc_mean']:.1%}" for vg in GUESS_GRID]
        print(f"{vs:.2f}        " + "  ".join(row))

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(summary, RESULTS_DIR / "summary.csv")
    fig_degradation(summary, RESULTS_DIR / "figure_degradation.pdf",
                    moteur_slip=PRUDENT_SLIP, moteur_guess=PRUDENT_GUESS,
                    title="Lot 3.6 — correction : moteur prudent (slip=0,20/guess=0,40)")
    fig_calibration(records, RESULTS_DIR / "figure_calibration.pdf")
    write_run_log(RESULTS_DIR / "run.log", len(records))
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure_degradation.pdf, "
         f"figure_calibration.pdf, run.log}}")

    fig_degradation(summary, ROOT / "lot3_6_degradation_figure.png",
                    moteur_slip=PRUDENT_SLIP, moteur_guess=PRUDENT_GUESS,
                    title="Lot 3.6 — correction : moteur prudent (slip=0,20/guess=0,40)")
    fig_calibration(records, ROOT / "lot3_6_calibration_figure.png")
    print("ecrit (copies racine) : lot3_6_degradation_figure.png, lot3_6_calibration_figure.png")


if __name__ == "__main__":
    main()
