"""
Action recommandee par docs/revue_d0.md (le test qui tranche vraiment) :
reconstruit une variante des reponses limitee a la PREMIERE tentative de
chaque (etudiant, exercice) -- problem_number == 1 -- et relance l'EM
dessus. Si guess retombe a une valeur plausible, D0 (repetition de
pratique) est la cause dominante et refaire A1 (option 2 de
PROMPT_A2_GRANULARITE.md) devient probablement inutile.

Ecrit dans data/diagnostics/, ne touche JAMAIS domain.yaml ni
responses.parquet (fichiers de production).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_junyi import aggregate_to_concepts, load_exercise_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_em  # reutilise sufficient_statistics / run_em / calibrate -- pas duplique

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
LOG_PATH = RAW_DIR / "junyi_ProblemLog_original.csv"
OUT_PATH = Path(__file__).resolve().parent / "responses_first_attempt.parquet"

CHUNK_SIZE = 1_000_000
USE_COLS = ["user_id", "exercise", "problem_number", "correct", "time_done"]

SCHEMA = pa.schema([
    ("student_id", pa.int64()),
    ("concept_id", pa.string()),
    ("correct", pa.bool_()),
    ("timestamp", pa.int64()),
])


def build_first_attempt_variant(log_path: Path = LOG_PATH, out_path: Path = OUT_PATH,
                                raw_dir: Path = RAW_DIR, chunk_size: int = CHUNK_SIZE
                                ) -> dict:
    """Meme pipeline que extract_responses.py, avec un filtre en plus :
    ne garder que problem_number == 1 (la toute premiere tentative de
    l'etudiant sur cet exercice, jamais une repetition de pratique)."""
    ex = load_exercise_table(raw_dir)
    name_to_concept, _ = aggregate_to_concepts(ex, edges=[])

    writer = None
    n_in = n_out = 0
    reader = pd.read_csv(log_path, usecols=USE_COLS,
                         dtype={"exercise": str, "correct": str},
                         chunksize=chunk_size)
    for chunk in reader:
        n_in += len(chunk)
        chunk = chunk[chunk["problem_number"] == 1]
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
    if writer is not None:
        writer.close()

    return {"lignes_brutes_lues": n_in, "lignes_premiere_tentative_retenues": n_out}


if __name__ == "__main__":
    stats = build_first_attempt_variant()
    print(f"Lignes brutes lues                          : {stats['lignes_brutes_lues']:,}")
    print(f"Lignes premiere-tentative retenues (mappees) : "
          f"{stats['lignes_premiere_tentative_retenues']:,}")
    print(f"Ecrit : {OUT_PATH}")
    print()

    print("--- Recalibration EM sur cette variante (5 restarts) ---")
    best, results, concepts, Z = calibrate_em.calibrate(
        responses_path=OUT_PATH, domain_path=calibrate_em.DOMAIN_PATH)

    print(f"Convergence : {best.converged}")
    finals = [r.log_likelihood_trace[-1] for r in results]
    print(f"log-vraisemblance finale (5 restarts) : {finals}")
    print()

    with open(calibrate_em.DOMAIN_PATH, encoding="utf-8") as f:
        current = {c["id"]: (c.get("slip"), c.get("guess"))
                  for c in yaml.safe_load(f)["concepts"]}

    print(f"{'concept':22s} {'slip':>7s} {'guess':>7s}   (rappel domain.yaml actuel)")
    for c, s, g in zip(concepts, best.slip, best.guess):
        slip_actuel, guess_actuel = current[c]
        print(f"  {c:20s} {s:7.3f} {g:7.3f}   (actuel : slip={slip_actuel}, guess={guess_actuel})")
