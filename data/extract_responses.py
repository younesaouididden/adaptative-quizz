"""
Tache A1 (suite) : construit responses.parquet a partir des logs bruts
Junyi15 (junyi_ProblemLog_original.csv, ~26M lignes, 2,6 Go), en reprojetant
chaque exercice sur son concept -- meme mapping que extract_junyi.py, pour
que domain.yaml et responses.parquet restent coherents entre eux.

Lecture PAR CHUNKS : le fichier est trop volumineux pour tenir masse en
memoire d'un coup (~26M lignes). Ecriture parquet incrementale (pyarrow
ParquetWriter) pour la meme raison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from extract_junyi import aggregate_to_concepts, load_exercise_table

RAW_DIR = Path(__file__).resolve().parent / "raw"
LOG_PATH = RAW_DIR / "junyi_ProblemLog_original.csv"
OUT_PATH = Path(__file__).resolve().parent / "responses.parquet"

CHUNK_SIZE = 1_000_000
USE_COLS = ["user_id", "exercise", "correct", "time_done"]

SCHEMA = pa.schema([
    ("student_id", pa.int64()),
    ("concept_id", pa.string()),
    ("correct", pa.bool_()),
    ("timestamp", pa.int64()),
])


def load_concept_mapping(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Meme mapping exercice -> concept que domain.yaml (extract_junyi.py) :
    reutilise aggregate_to_concepts, sans edges (pas necessaires pour le
    mapping lui-meme, seulement pour les comptages de prerequis)."""
    ex = load_exercise_table(raw_dir)
    name_to_concept, _ = aggregate_to_concepts(ex, edges=[])
    return name_to_concept


def process_chunk(chunk: pd.DataFrame, name_to_concept: dict[str, str]) -> pd.DataFrame:
    """Filtre sur les exercices mappes (les autres -- biology, orphelins --
    sont exclus, comme dans domain.yaml) et reprojette sur les concepts.

    'correct' est 'true'/'false' en minuscules dans le CSV source : converti
    explicitement plutot que de compter sur l'inference de type de pandas
    (qui ne reconnait pas toujours ce format comme booleen).
    """
    concept_id = chunk["exercise"].map(name_to_concept)
    mask = concept_id.notna()
    return pd.DataFrame({
        "student_id": chunk.loc[mask, "user_id"].astype("int64"),
        "concept_id": concept_id[mask].astype(str),
        "correct": chunk.loc[mask, "correct"].str.lower().map({"true": True, "false": False}),
        "timestamp": chunk.loc[mask, "time_done"].astype("int64"),
    })


def extract(log_path: Path = LOG_PATH, out_path: Path = OUT_PATH,
           raw_dir: Path = RAW_DIR, chunk_size: int = CHUNK_SIZE) -> dict:
    name_to_concept = load_concept_mapping(raw_dir)

    writer = None
    n_in = n_out = 0
    try:
        reader = pd.read_csv(log_path, usecols=USE_COLS,
                             dtype={"exercise": str, "correct": str},
                             chunksize=chunk_size)
        for chunk in reader:
            n_in += len(chunk)
            out = process_chunk(chunk, name_to_concept)
            n_out += len(out)
            if len(out) == 0:
                continue
            table = pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, SCHEMA)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return {
        "lignes_lues": n_in,
        "lignes_retenues": n_out,
        "taux_retenu": (n_out / n_in) if n_in else 0.0,
    }


if __name__ == "__main__":
    stats = extract()
    print(f"Lignes lues     : {stats['lignes_lues']:,}")
    print(f"Lignes retenues : {stats['lignes_retenues']:,}  "
          f"({100 * stats['taux_retenu']:.1f}%)")
    print(f"Ecrit : {OUT_PATH}")
