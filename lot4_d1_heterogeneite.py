"""Lot 4 -- D1, test direct de l'hypothese d'heterogeneite (plan_action_code.md).

Ne re-teste PAS tout le domaine piste A : prend le SEUL bucket arithmetic
(301 exercices, area Junyi brute), le subdivise comme en piste B
(arithmetic_base / fractions_ratios, meme mapping topic->concept que la
premiere tentative a 9 concepts -- cf. git history de data/extract_junyi.py,
commit 568dc1d), et relance l'EM ISOLEMENT sur ce sous-domaine a 2 concepts
seul (pas joint avec les 7 ou 9 autres concepts comme dans les tentatives
precedentes) -- pour savoir si guess baisse UNE FOIS LA SUBDIVISION FAITE,
independamment de toute interference avec le reste du domaine.

Si oui : l'hypothese d'heterogeneite passe d'une explication verbale a une
preuve chiffree, et la subdivision a 7 concepts de piste B se trouve
justifiee a posteriori (meme si piste B n'est pas elle-meme calibree).

Si non : le probleme n'est pas (seulement) la granularite, cf. §4 point 4
de CONTEXTE_PROJET.md (le jeu complet reste bien identifie et affiche quand
meme un guess eleve) -- a documenter comme limite non resolue, pas a forcer.

Etape 1 -- extraction (reutilise le chunking de data/extract_responses.py,
necessaire car responses.parquet a deja agrege sur 'arithmetic', on ne peut
plus en retrouver le topic individuel -- il faut retraiter le log brut) :
filtre junyi_ProblemLog_original.csv aux SEULS exercices de l'area
'arithmetic', reprojette sur (arithmetic_base, fractions_ratios).

Etape 2 -- calibration EM isolee sur ce sous-domaine (reutilise
data/calibrate_em.py : calibrate(), write_calibrated_domain(), en pointant
sur ce domaine/ces reponses au lieu des fichiers principaux -- AUCUNE
modification de data/domain.yaml ni data/responses.parquet).

Usage : python lot4_d1_heterogeneite.py
Ecrit results/lot4_d1_heterogeneite/{domain_d1.yaml, responses_d1.parquet,
run.log}.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
EXERCISE_TABLE = RAW_DIR / "junyi_Exercise_table.csv"
LOG_PATH = RAW_DIR / "junyi_ProblemLog_original.csv"
RESULTS_DIR = ROOT / "results" / "lot4_d1_heterogeneite"
DOMAIN_D1_PATH = RESULTS_DIR / "domain_d1.yaml"
RESPONSES_D1_PATH = RESULTS_DIR / "responses_d1.parquet"

sys.path.insert(0, str(ROOT / "data"))
from calibrate_em import calibrate, write_calibrated_domain  # noqa: E402

CHUNK_SIZE = 1_000_000
USE_COLS = ["user_id", "exercise", "correct", "time_done"]
SCHEMA = pa.schema([
    ("student_id", pa.int64()),
    ("concept_id", pa.string()),
    ("correct", pa.bool_()),
    ("timestamp", pa.int64()),
])

# Meme mapping que la premiere tentative a 9 concepts (git commit 568dc1d,
# data/extract_junyi.py) -- seul le sous-ensemble pertinent pour l'area
# 'arithmetic' (301 exercices, 12 topics, couverture complete verifiee).
ARITHMETIC_TOPIC_OVERRIDES = {
    "addition-subtraction": "arithmetic_base", "multiplication-division": "arithmetic_base",
    "order-of-operations": "arithmetic_base", "quantity-sense": "arithmetic_base",
    "telling-time": "arithmetic_base", "factors-multiples": "arithmetic_base",
    "fractions": "fractions_ratios", "decimals": "fractions_ratios",
    "ratio-percentage": "fractions_ratios", "rates-and-ratios": "fractions_ratios",
    "unit-conversion": "fractions_ratios", "sequence_and_series": "fractions_ratios",
}


def build_exercise_mapping() -> dict[str, str]:
    ex = pd.read_csv(EXERCISE_TABLE)
    arithmetic = ex[ex["area"] == "arithmetic"]
    missing = set(arithmetic["topic"]) - set(ARITHMETIC_TOPIC_OVERRIDES)
    if missing:
        raise ValueError(f"topics arithmetic non couverts par le mapping : {missing}")
    return dict(zip(arithmetic["name"], arithmetic["topic"].map(ARITHMETIC_TOPIC_OVERRIDES)))


def write_domain_d1(path: Path) -> None:
    data = {
        "schema_version": 1,
        "domain": "junyi_arithmetic_d1",
        "description": "Sous-domaine Lot 4/D1 : bucket arithmetic (piste A) "
                       "subdivise comme piste B, calibre isolement.",
        "concepts": [
            {"id": "arithmetic_base", "label": "Arithmetique de base", "calibrated": False},
            {"id": "fractions_ratios", "label": "Fractions et ratios", "calibrated": False},
        ],
        "prerequisites": [["arithmetic_base", "fractions_ratios"]],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def extract_d1_responses(name_to_concept: dict[str, str], out_path: Path) -> dict:
    writer = None
    n_in = n_out = 0
    try:
        reader = pd.read_csv(LOG_PATH, usecols=USE_COLS,
                             dtype={"exercise": str, "correct": str},
                             chunksize=CHUNK_SIZE)
        for chunk in reader:
            n_in += len(chunk)
            concept_id = chunk["exercise"].map(name_to_concept)
            mask = concept_id.notna()
            out = pd.DataFrame({
                "student_id": chunk.loc[mask, "user_id"].astype("int64"),
                "concept_id": concept_id[mask].astype(str),
                "correct": chunk.loc[mask, "correct"].str.lower().map({"true": True, "false": False}),
                "timestamp": chunk.loc[mask, "time_done"].astype("int64"),
            })
            n_out += len(out)
            if len(out) == 0:
                continue
            table = pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, SCHEMA)
            writer.write_table(table)
            print(f"  ... {n_in:,} lignes lues, {n_out:,} retenues", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return {"lignes_lues": n_in, "lignes_retenues": n_out}


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "inconnu (git indisponible)"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Etape 1/2 -- extraction des reponses arithmetic (subdivisees)...")
    name_to_concept = build_exercise_mapping()
    print(f"  {len(name_to_concept)} exercices mappes "
         f"({sum(1 for v in name_to_concept.values() if v == 'arithmetic_base')} "
         f"arithmetic_base, "
         f"{sum(1 for v in name_to_concept.values() if v == 'fractions_ratios')} "
         f"fractions_ratios)")
    write_domain_d1(DOMAIN_D1_PATH)

    if RESPONSES_D1_PATH.exists():
        print(f"  {RESPONSES_D1_PATH} existe deja, extraction sautee (supprimer pour refaire)")
        stats = {"lignes_lues": None, "lignes_retenues": None}
    else:
        stats = extract_d1_responses(name_to_concept, RESPONSES_D1_PATH)
        print(f"  lignes lues : {stats['lignes_lues']:,}  "
             f"retenues : {stats['lignes_retenues']:,}")

    print("\nEtape 2/2 -- calibration EM isolee (5 redemarrages)...")
    best, results, concepts, Z = calibrate(
        n_restarts=5, seed=0, domain_path=DOMAIN_D1_PATH, responses_path=RESPONSES_D1_PATH)

    print(f"  convergence : {best.converged} ({len(best.log_likelihood_trace)} iterations)")
    slips = np.array([r.slip for r in results])
    guesses = np.array([r.guess for r in results])
    print(f"  ecart-type slip (5 redemarrages)  : {slips.std(axis=0).round(4)}")
    print(f"  ecart-type guess (5 redemarrages) : {guesses.std(axis=0).round(4)}")
    print()
    print(f"{'concept':20s} {'slip':>7s} {'guess':>7s}")
    for c, s, g in zip(concepts, best.slip, best.guess):
        print(f"  {c:18s} {s:7.4f} {g:7.4f}")

    write_calibrated_domain(concepts, best.slip, best.guess, domain_path=DOMAIN_D1_PATH)

    with open(RESULTS_DIR / "run.log", "w", encoding="utf-8") as f:
        f.write(f"date (UTC)   : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande     : python lot4_d1_heterogeneite.py\n")
        f.write(f"commit git   : {_git_commit()}\n")
        f.write(f"exercices    : {len(name_to_concept)} (area arithmetic, junyi_Exercise_table.csv)\n")
        f.write(f"reponses lues/retenues : {stats['lignes_lues']}/{stats['lignes_retenues']}\n")
        f.write(f"graine EM    : seed=0, 5 redemarrages independants\n")
        f.write(f"convergence  : {best.converged}\n\n")
        f.write("Comparaison (le point de D1) :\n")
        f.write("  arithmetic combine (piste A, 5 concepts)      : slip=0.1064 guess=0.7462\n")
        f.write("  arithmetic_base (9 concepts, joint, ancien)   : slip=0.096  guess=0.756\n")
        f.write("  fractions_ratios (9 concepts, joint, ancien)  : slip=0.141  guess=0.685\n")
        f.write(f"  arithmetic_base (D1, ISOLE)                   : slip={best.slip[concepts.index('arithmetic_base')]:.4f} "
               f"guess={best.guess[concepts.index('arithmetic_base')]:.4f}\n")
        f.write(f"  fractions_ratios (D1, ISOLE)                  : slip={best.slip[concepts.index('fractions_ratios')]:.4f} "
               f"guess={best.guess[concepts.index('fractions_ratios')]:.4f}\n")

    print(f"\necrit : {DOMAIN_D1_PATH}, {RESPONSES_D1_PATH}, {RESULTS_DIR}/run.log")


if __name__ == "__main__":
    main()
