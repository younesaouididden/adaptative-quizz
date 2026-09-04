"""Lot 4 -- A3 sur le domaine calibre piste A (plan_action_code.md).

Rejoue le benchmark A3 (adaptatif / aleatoire / CAT-IRT) sur le domaine
REELLEMENT calibre par EM sur les logs Junyi (data/domain.yaml, 5 concepts,
|Z|=18, guess 0.50-0.75) au lieu de piste B (guess borne a 0.25).

Attendu (plan, relie au Lot 2.4) : degradation forte du gain d'efficacite --
deja predite quantitativement par expected_fisher_rao_step et
item_information (Lot 1-2), ici mesuree directement sur un vrai benchmark
de bout en bout.

Plafond de questions releve a 150 (pas 30) : cf. note_calibration.md §6,
avec guess=0.75 la formule fermee predit jusqu'a ~180 questions pour un
seul concept -- sous 30, on mesurerait le plafond, pas la methode.

Banque synthetique : data/domain.yaml n'a pas de banque de questions
individuelles (calibration EM au niveau concept, pas item). 40 questions
par concept, TOUTES partageant le (slip, guess) calibre du concept
(honnete : aucune variation inventee au niveau item, on n'a pas cette
donnee) -- juste assez pour que l'argmax ait de quoi departager au-dela
du premier tour par concept (cf. kst_engine.pi_star, meme raisonnement
que le refactor concept/question de Sprint 0).

Usage : python lot4_a3_piste_a.py
Ecrit results/lot4_a3_piste_a/{raw.csv, summary.csv, figure.pdf, run.log}
+ une copie lot4_a3_piste_a_figure.png a la racine.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from benchmark_a3 import concept_accuracy, run_benchmark, summarize
from kst_engine import Concept, Domain, Question, item_information, questions_needed

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "data" / "domain.yaml"
RESULTS_DIR = ROOT / "results" / "lot4_a3_piste_a"

QUESTIONS_PER_CONCEPT = 40
MAX_QUESTIONS = 150

POLICY_LABELS = {
    "adaptatif": "Adaptatif (π*)\n(gain d'info, KST)",
    "aleatoire": "Aléatoire",
    "cat_irt": "CAT-IRT\n(baseline 3PL)",
}
POLICY_COLORS = {"adaptatif": "#2E86AB", "aleatoire": "#C0C0C0", "cat_irt": "#C0504D"}


def load_piste_a_bank(path: Path, n_per_concept: int) -> tuple[Domain, list[dict]]:
    """Construit une banque synthetique (n_per_concept questions identiques
    par concept, slip/guess REELLEMENT calibres par EM) a partir de
    data/domain.yaml, qui ne contient que des parametres au niveau concept."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    concepts = [Concept(c["id"]) for c in data["concepts"]]
    prereqs = [tuple(p) for p in data["prerequisites"]]
    slip_guess = {c["id"]: (c["slip"], c["guess"]) for c in data["concepts"]}

    questions, meta = [], []
    for c in data["concepts"]:
        slip, guess = slip_guess[c["id"]]
        for i in range(n_per_concept):
            questions.append(Question(f"{c['id']}_{i + 1}", c["id"], slip=slip, guess=guess))
            meta.append({"difficulty": 2})   # pas de donnee de difficulte par item en piste A

    domain = Domain(concepts=concepts, prereqs=prereqs, questions=questions)
    return domain, meta


def predicted_questions(path: Path) -> float:
    """Prediction fermee (Lot 1, questions_needed/item_information) : somme
    sur les concepts de log_odds/item_information(slip_i, guess_i), puisque
    piste A n'a pas un (slip, guess) uniforme comme piste B."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    log_odds = np.log(0.95 / 0.05)
    return sum(log_odds / item_information(c["slip"], c["guess"]) for c in data["concepts"])


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "seed", "z_true", "n_questions",
                                          "correct_diagnosis", "concept_accuracy"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(summary: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["politique", "n_questions_moyen", "n_questions_std",
                   "exactitude_etat_exact", "exactitude_par_concept"])
        for policy, s in summary.items():
            w.writerow([policy, f"{s['n_questions_mean']:.2f}", f"{s['n_questions_std']:.2f}",
                       f"{s['exact_match_rate']:.3f}", f"{s['concept_accuracy_mean']:.3f}"])


def fig_summary(summary: dict, path: Path) -> None:
    policies = list(POLICY_LABELS.keys())
    n_q = [summary[p]["n_questions_mean"] for p in policies]
    acc = [summary[p]["concept_accuracy_mean"] * 100 for p in policies]
    colors = [POLICY_COLORS[p] for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    b1 = ax1.bar([POLICY_LABELS[p] for p in policies], n_q, color=colors)
    ax1.set_ylabel("Nombre moyen de questions")
    ax1.set_title(f"Questions pour atteindre l'arrêt (plafond={MAX_QUESTIONS})")
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

    fig.suptitle("Lot 4 — Benchmark A3 sur piste A (domaine calibré EM, 5 concepts, |Z|=18)")
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


def write_run_log(path: Path, n_states: int, predicted: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)         : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande           : python lot4_a3_piste_a.py\n")
        f.write(f"commit git         : {_git_commit()}\n")
        f.write(f"domaine            : {DOMAIN_PATH.relative_to(ROOT)} (piste A, calibre EM)\n")
        f.write(f"questions/concept  : {QUESTIONS_PER_CONCEPT} (synthetiques, slip/guess reels partages)\n")
        f.write(f"plafond questions  : {MAX_QUESTIONS} (cf. note_calibration.md §6)\n")
        f.write(f"graines            : seed = index de l'etat dans domain.Z, 0..{n_states - 1}\n")
        f.write(f"prediction fermee  : {predicted:.1f} questions (adaptatif, "
               f"somme par concept de log(19)/item_information)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta = load_piste_a_bank(DOMAIN_PATH, QUESTIONS_PER_CONCEPT)
    print(f"domaine piste A : {domain.n_states} etats, {domain.n_questions} questions")

    predicted = predicted_questions(DOMAIN_PATH)
    print(f"prediction fermee (questions_needed par concept) : {predicted:.1f} questions")

    records = run_benchmark(domain, meta, max_questions=MAX_QUESTIONS)
    summary = summarize(records)

    for policy, s in summary.items():
        print(f"{policy:10s} n_questions={s['n_questions_mean']:6.2f}±{s['n_questions_std']:.2f}  "
             f"exact={s['exact_match_rate']:.1%}  concept_acc={s['concept_accuracy_mean']:.1%}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(summary, RESULTS_DIR / "summary.csv")
    fig_summary(summary, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", len(domain.Z), predicted)
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    fig_summary(summary, ROOT / "lot4_a3_piste_a_figure.png")
    print("ecrit (copie racine) : lot4_a3_piste_a_figure.png")


if __name__ == "__main__":
    main()
