"""Lot 1, E1 -- fidelite de la politique approximee (plan_action_code.md).

Sur des croyances p ISSUES DE VRAIES TRAJECTOIRES adaptatives (pas de prior
uniforme artificiel -- pi_star pilote la trajectoire, on capture p AVANT
chaque decision), pour chaque N in {1,3,5,10,30,100} et chaque mode
(sample_y, sample_z), mesure :
  - taux d'accord : argmax(pi_hat) == argmax(pi_star)
  - regret en IG   : IG_exact(a_MC) / IG_exact(a*)

Le taux d'accord seul est trompeur (choisir la 2e meilleure question quasi
equivalente ne coute presque rien) -- le regret est la metrique honnete
(cf. plan_action_code.md, Lot 1.2).

Usage : python lot1_e1_fidelite_politique.py
Ecrit results/lot1_e1/{raw.csv, summary.csv, figure.pdf, run.log} (source
canonique, Lot 0) + une copie lot1_e1_figure.png a la racine (commodite pour
RAPPORT_AVANCEMENT.md, qui l'embarque -- meme pattern que benchmark_a3.py).
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domains.loader import load_domain_yaml
from kst_engine import (
    Domain,
    bayes_update,
    information_gain_exact,
    pi_hat,
    pi_star,
    should_stop,
    uniform_prior,
)

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot1_e1"

N_VALUES = [1, 3, 5, 10, 30, 100]
MODES = ["sample_y", "sample_z"]
MODE_IDX = {"sample_y": 0, "sample_z": 1}
N_STATES_SAMPLED = 20   # sous-echantillon de domain.Z pour generer les trajectoires
N_REPS = 10             # replications MC independantes par (point, mode, N)
MASTER_SEED = 0         # graine racine de l'echantillonnage des etats


def collect_trajectory_beliefs(domain: Domain, z_true: frozenset,
                               seed: int, max_questions: int = 30
                               ) -> list[tuple[np.ndarray, set]]:
    """Rejoue une vraie trajectoire adaptative (pi_star, meme logique que
    kst_engine.simulate) et retourne la sequence de (p, asked) rencontree
    AVANT chaque decision -- ce sont les croyances "reelles" d'E1."""
    rng = np.random.default_rng(seed)
    p = uniform_prior(domain)
    asked: set[int] = set()
    points = [(p.copy(), set(asked))]

    while True:
        q_best, ig = pi_star(p, domain, asked)
        if should_stop(p, domain, asked, max_questions=max_questions, ig=ig):
            break
        question = domain.questions[q_best]
        # verite terrain BLIM, comme kst_engine.simulate
        true_p = (1 - question.slip) if question.concept in z_true else question.guess
        correct = bool(rng.random() < true_p)
        p = bayes_update(p, domain, q_best, correct)
        asked.add(q_best)
        points.append((p.copy(), set(asked)))
    return points


def run_e1(domain: Domain) -> list[dict]:
    all_z = list(domain.Z)
    rng_pick = np.random.default_rng(MASTER_SEED)
    n_pick = min(N_STATES_SAMPLED, len(all_z))
    sampled_idx = rng_pick.choice(len(all_z), size=n_pick, replace=False)

    belief_points: list[tuple[np.ndarray, set]] = []
    for i in sampled_idx:
        belief_points.extend(collect_trajectory_beliefs(domain, all_z[int(i)], seed=int(i)))

    # argmax trivial/indefini s'il reste <2 questions -- on les exclut
    belief_points = [(p, asked) for p, asked in belief_points
                     if domain.n_questions - len(asked) >= 2]

    records = []
    for point_id, (p, asked) in enumerate(belief_points):
        q_star, _ = pi_star(p, domain, asked)
        exact_star = information_gain_exact(p, domain, q_star)
        if exact_star <= 1e-9:
            continue  # point deja quasi-certain, le regret n'est pas defini

        for mode in MODES:
            for n in N_VALUES:
                for rep in range(N_REPS):
                    seed_seq = np.random.SeedSequence([point_id, MODE_IDX[mode], n, rep])
                    rng = np.random.default_rng(seed_seq)
                    q_hat, _ = pi_hat(p, domain, asked, n_samples=n, mode=mode, rng=rng)
                    exact_hat = information_gain_exact(p, domain, q_hat)
                    records.append({
                        "point_id": point_id,
                        "n_remaining": domain.n_questions - len(asked),
                        "mode": mode,
                        "n_samples": n,
                        "rep": rep,
                        "agree": q_hat == q_star,
                        "regret": exact_hat / exact_star,
                    })
    return records


def summarize(records: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    summary = {}
    for mode in MODES:
        for n in N_VALUES:
            rows = [r for r in records if r["mode"] == mode and r["n_samples"] == n]
            agree = np.array([r["agree"] for r in rows], dtype=float)
            regret = np.array([r["regret"] for r in rows], dtype=float)
            summary[(mode, n)] = {
                "agreement_rate": float(agree.mean()),
                "regret_mean": float(regret.mean()),
                "regret_p10": float(np.percentile(regret, 10)),
                "n_obs": len(rows),
            }
    return summary


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["point_id", "n_remaining", "mode",
                                          "n_samples", "rep", "agree", "regret"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(summary: dict[tuple[str, int], dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_samples", "taux_accord", "regret_moyen",
                   "regret_p10", "n_observations"])
        for (mode, n), s in sorted(summary.items()):
            w.writerow([mode, n, f"{s['agreement_rate']:.4f}", f"{s['regret_mean']:.4f}",
                       f"{s['regret_p10']:.4f}", s["n_obs"]])


def fig_summary(summary: dict[tuple[str, int], dict[str, float]], path: Path) -> None:
    colors = {"sample_y": "#2E86AB", "sample_z": "#C0504D"}
    markers = {"sample_y": "o", "sample_z": "s"}
    labels = {"sample_y": "sample_y (échantillonne les réponses)",
             "sample_z": "sample_z (échantillonne les états)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for mode in MODES:
        agree = [summary[(mode, n)]["agreement_rate"] * 100 for n in N_VALUES]
        regret = [summary[(mode, n)]["regret_mean"] * 100 for n in N_VALUES]
        ax1.plot(N_VALUES, agree, marker=markers[mode], color=colors[mode],
                linewidth=2, label=labels[mode])
        ax2.plot(N_VALUES, regret, marker=markers[mode], color=colors[mode],
                linewidth=2, label=labels[mode])

    ax1.set_xscale("log")
    ax1.set_xlabel("N (échantillons Monte Carlo)")
    ax1.set_ylabel("Taux d'accord argmax(π̂_N) = argmax(π*) (%)")
    ax1.set_title("Fidélité de la politique")
    ax1.set_ylim(0, 105)
    ax1.axhline(100, color="black", linewidth=0.5, linestyle="--")

    ax2.set_xscale("log")
    ax2.set_xlabel("N (échantillons Monte Carlo)")
    ax2.set_ylabel("Regret moyen IG_exact(π̂_N) / IG_exact(π*) (%)")
    ax2.set_title("Regret en gain d'information\n(métrique honnête — cf. plan_action_code.md 1.2)")
    ax2.set_ylim(0, 105)
    ax2.axhline(100, color="black", linewidth=0.5, linestyle="--")

    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Lot 1 / E1 — fidélité de π̂_N sur des croyances issues de vraies trajectoires "
                 "(piste B)")
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


def write_run_log(path: Path, n_points: int, n_records: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)       : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande         : python lot1_e1_fidelite_politique.py\n")
        f.write(f"commit git       : {_git_commit()}\n")
        f.write(f"domaine          : {DOMAIN_PATH.relative_to(ROOT)}\n")
        f.write(f"graine racine    : MASTER_SEED={MASTER_SEED} "
               f"(choix des {N_STATES_SAMPLED} etats + trajectoires)\n")
        f.write(f"graines MC       : np.random.SeedSequence([point_id, mode_idx, n, rep]), "
               f"deterministe par (point, mode, N, replication)\n")
        f.write(f"points de croyance : {n_points} (issus de {N_STATES_SAMPLED} trajectoires)\n")
        f.write(f"lignes brutes    : {n_records} "
               f"({n_points} points x {len(MODES)} modes x {len(N_VALUES)} N x {N_REPS} reps)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)

    records = run_e1(domain)
    summary = summarize(records)
    n_points = len({r["point_id"] for r in records})

    print(f"points de croyance retenus : {n_points}")
    for mode in MODES:
        print(f"\n{mode} :")
        for n in N_VALUES:
            s = summary[(mode, n)]
            print(f"  N={n:4d}  accord={s['agreement_rate']:.1%}  "
                 f"regret_moyen={s['regret_mean']:.1%}  regret_p10={s['regret_p10']:.1%}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(summary, RESULTS_DIR / "summary.csv")
    fig_summary(summary, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", n_points, len(records))
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    fig_summary(summary, ROOT / "lot1_e1_figure.png")
    print("ecrit (copie racine) : lot1_e1_figure.png")


if __name__ == "__main__":
    main()
