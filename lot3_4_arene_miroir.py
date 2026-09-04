"""Lot 3.4 -- arene miroir (plan_action_code.md).

Genere les reponses depuis un 3PL a theta continu (le modele DE L'IRT, cf.
irt_baseline.py) au lieu du BLIM discret, et fait tourner les deux
politiques (KST adaptatif, CAT-IRT) sur cette meme verite generative.

Message attendu (et voulu) : contrairement au benchmark A3 (verite BLIM,
ou KST domine, cf. RAPPORT_LOT... non, benchmark_a3.py), ici l'IRT doit
dominer -- chaque modele gagne sur SA PROPRE verite generative. Ce n'est
pas une contradiction du benchmark A3, c'est la preuve que ni l'un ni
l'autre resultat n'est truque : la vraie question n'est pas "quel
algorithme est meilleur dans l'absolu" mais "lequel des deux modeles
generatifs (KST discret ou IRT continu) decrit le mieux les donnees reelles
Junyi" -- ce qui renvoie a la piste A, pas a ce benchmark.

Etat de maitrise "vrai" derive du theta continu : meme convention que
simulate_irt (irt_baseline.py), maitrise ssi theta >= b moyen des items du
concept -- necessaire pour comparer un diagnostic multi-concept (KST) a un
theta scalaire (IRT) sur un terrain commun.

Usage : python lot3_4_arene_miroir.py
Ecrit results/lot3_4_arene_miroir/{raw.csv, summary.csv, figure.pdf, run.log}
+ une copie lot3_4_arene_miroir_figure.png a la racine.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_a3 import concept_accuracy
from domains.loader import load_domain_yaml
from irt_baseline import build_irt_items, eap_theta, p_correct_3pl, select_next_irt
from kst_engine import Domain, bayes_update, pi_star, should_stop, uniform_prior

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot3_4_arene_miroir"

N_STUDENTS = 200
MASTER_SEED = 0


def z_from_theta(theta: float, items: list[dict], domain: Domain) -> frozenset:
    """Etat de maitrise 'vrai' derive d'un theta continu -- meme convention
    que simulate_irt : concept maitrise ssi theta >= b moyen de ses items."""
    concept_bs: dict[str, list[float]] = {}
    for q, it in zip(domain.questions, items):
        concept_bs.setdefault(q.concept, []).append(it["b"])
    return frozenset(c for c, bs in concept_bs.items() if theta >= np.mean(bs))


def simulate_kst_on_theta(domain: Domain, theta_true: float, items: list[dict],
                          seed: int = 0, max_questions: int = 30) -> dict:
    """Politique KST (pi_star), mais la reponse de l'etudiant est generee
    par le 3PL (theta_true), pas par le BLIM -- meme moteur de selection
    que simulate(), verite generative differente."""
    rng = np.random.default_rng(seed)
    p = uniform_prior(domain)
    asked: set[int] = set()

    while True:
        q_best, ig = pi_star(p, domain, asked)
        if should_stop(p, domain, asked, max_questions=max_questions, ig=ig):
            break
        it = items[q_best]
        true_p = p_correct_3pl(theta_true, it["a"], it["b"], it["c"])
        correct = bool(rng.random() < true_p)
        p = bayes_update(p, domain, q_best, correct)
        asked.add(q_best)

    z_hat = domain.Z[int(np.argmax(p))]
    return {"n_questions": len(asked), "z_hat": z_hat}


def simulate_irt_on_theta(domain: Domain, items: list[dict], theta_true: float,
                          seed: int = 0, max_questions: int = 30,
                          se_stop: float = 0.3) -> dict:
    """Politique CAT-IRT sur sa PROPRE verite generative (3PL) -- meme
    moteur de selection que simulate_irt (irt_baseline.py)."""
    rng = np.random.default_rng(seed)
    responses: list[tuple[float, float, float, bool]] = []
    asked: set[int] = set()
    theta_hat, se = eap_theta(responses)

    while True:
        if len(asked) >= min(max_questions, len(items)) or se <= se_stop:
            break
        q, _ = select_next_irt(theta_hat, items, asked)
        if q is None:
            break
        it = items[q]
        true_p = p_correct_3pl(theta_true, it["a"], it["b"], it["c"])
        correct = bool(rng.random() < true_p)
        responses.append((it["a"], it["b"], it["c"], correct))
        asked.add(q)
        theta_hat, se = eap_theta(responses)

    concept_bs: dict[str, list[float]] = {}
    for q, it in zip(domain.questions, items):
        concept_bs.setdefault(q.concept, []).append(it["b"])
    z_hat = frozenset(c for c, bs in concept_bs.items() if theta_hat >= np.mean(bs))
    return {"n_questions": len(asked), "z_hat": z_hat, "theta_hat": theta_hat}


def run(domain: Domain, meta: list[dict]) -> list[dict]:
    items = build_irt_items(domain, meta)
    concepts = [c.name for c in domain.concepts]
    rng = np.random.default_rng(MASTER_SEED)
    thetas = rng.normal(0.0, 1.0, size=N_STUDENTS)   # meme prior que l'EAP

    records = []
    for i, theta_true in enumerate(thetas):
        z_true = z_from_theta(float(theta_true), items, domain)
        r_kst = simulate_kst_on_theta(domain, float(theta_true), items, seed=i)
        r_irt = simulate_irt_on_theta(domain, items, float(theta_true), seed=i)
        for policy, r in (("kst_adaptatif", r_kst), ("cat_irt", r_irt)):
            records.append({
                "student": i, "theta_true": float(theta_true),
                "policy": policy, "n_questions": r["n_questions"],
                "concept_accuracy": concept_accuracy(r["z_hat"], z_true, concepts),
            })
    return records


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["student", "theta_true", "policy",
                                          "n_questions", "concept_accuracy"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n_questions_moyen", "n_questions_std",
                   "exactitude_par_concept", "exactitude_par_concept_std"])
        for policy in ("kst_adaptatif", "cat_irt"):
            rows = [r for r in records if r["policy"] == policy]
            n_q = np.array([r["n_questions"] for r in rows], dtype=float)
            acc = np.array([r["concept_accuracy"] for r in rows], dtype=float)
            w.writerow([policy, f"{n_q.mean():.2f}", f"{n_q.std():.2f}",
                       f"{acc.mean():.3f}", f"{acc.std():.3f}"])


def fig_summary(records: list[dict], path: Path) -> None:
    colors = {"kst_adaptatif": "#2E86AB", "cat_irt": "#C0504D"}
    labels = {"kst_adaptatif": "KST adaptatif\n(mal spécifié ici)", "cat_irt": "CAT-IRT\n(sur sa propre vérité)"}
    policies = ["kst_adaptatif", "cat_irt"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    n_q = [np.mean([r["n_questions"] for r in records if r["policy"] == p]) for p in policies]
    acc = [np.mean([r["concept_accuracy"] for r in records if r["policy"] == p]) * 100
          for p in policies]

    b1 = ax1.bar([labels[p] for p in policies], n_q, color=[colors[p] for p in policies])
    ax1.set_ylabel("Nombre moyen de questions")
    ax1.set_title("Questions pour atteindre l'arrêt")
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center")

    b2 = ax2.bar([labels[p] for p in policies], acc, color=[colors[p] for p in policies])
    ax2.set_ylabel("Exactitude par concept (%)")
    ax2.set_title("Précision du diagnostic")
    ax2.set_ylim(0, 100)
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Lot 3.4 — arène miroir : vérité générative 3PL (le modèle DE l'IRT)\n"
                f"({N_STUDENTS} étudiants simulés, θ ~ N(0,1))")
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


def write_run_log(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)   : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande     : python lot3_4_arene_miroir.py\n")
        f.write(f"commit git   : {_git_commit()}\n")
        f.write(f"domaine      : {DOMAIN_PATH.relative_to(ROOT)}\n")
        f.write(f"n_etudiants  : {N_STUDENTS}\n")
        f.write(f"graines      : MASTER_SEED={MASTER_SEED} pour theta~N(0,1) ; "
               f"seed=i (0..{N_STUDENTS - 1}) pour chaque etudiant, "
               f"meme graine pour les deux politiques (comparaison appariee)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)

    records = run(domain, meta)
    for policy in ("kst_adaptatif", "cat_irt"):
        rows = [r for r in records if r["policy"] == policy]
        n_q = np.mean([r["n_questions"] for r in rows])
        acc = np.mean([r["concept_accuracy"] for r in rows])
        print(f"{policy:16s} n_questions={n_q:.2f}  concept_accuracy={acc:.1%}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(records, RESULTS_DIR / "summary.csv")
    fig_summary(records, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log")
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    fig_summary(records, ROOT / "lot3_4_arene_miroir_figure.png")
    print("ecrit (copie racine) : lot3_4_arene_miroir_figure.png")


if __name__ == "__main__":
    main()
