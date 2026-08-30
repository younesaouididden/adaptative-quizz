"""
Suite pytest de extract_responses.py -- fixtures synthetiques (petit CSV
fabrique a la main), jamais le fichier reel de 26M lignes.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from extract_responses import extract, process_chunk


@pytest.fixture
def name_to_concept():
    return {"ar1": "arithmetic_base", "ar2": "arithmetic_base", "ge1": "geometry"}


class TestProcessChunk:

    def test_mappe_exercice_vers_concept(self, name_to_concept):
        chunk = pd.DataFrame({
            "user_id": [1, 2], "exercise": ["ar1", "ge1"],
            "correct": ["true", "false"], "time_done": [1000, 2000],
        })
        out = process_chunk(chunk, name_to_concept)
        assert out["concept_id"].tolist() == ["arithmetic_base", "geometry"]
        assert out["correct"].tolist() == [True, False]
        assert out["student_id"].tolist() == [1, 2]
        assert out["timestamp"].tolist() == [1000, 2000]

    def test_exercice_non_mappe_exclu(self, name_to_concept):
        """Un exercice hors du domaine retenu (ex. 'biology') ne doit pas
        se retrouver dans les reponses, comme dans domain.yaml."""
        chunk = pd.DataFrame({
            "user_id": [1, 2], "exercise": ["ar1", "bio_exercise_inconnu"],
            "correct": ["true", "true"], "time_done": [1000, 2000],
        })
        out = process_chunk(chunk, name_to_concept)
        assert len(out) == 1
        assert out["concept_id"].tolist() == ["arithmetic_base"]

    def test_correct_minuscule_converti_en_bool(self, name_to_concept):
        chunk = pd.DataFrame({
            "user_id": [1], "exercise": ["ar1"],
            "correct": ["TRUE"], "time_done": [1000],  # variante de casse
        })
        out = process_chunk(chunk, name_to_concept)
        assert out["correct"].tolist() == [True]
        assert out["correct"].dtype == bool


class TestExtract:

    def test_extraction_bout_en_bout(self, tmp_path):
        """Petit CSV synthetique + petite table d'exercices -> parquet
        correct, sans toucher au fichier Junyi reel."""
        raw_dir = tmp_path
        ex = pd.DataFrame({
            "name": ["ar1", "ge1", "bi1"],
            "area": ["arithmetic", "geometry", "biology"],
            "topic": ["addition-subtraction", "basic-geometry", "x"],
            "prerequisites": [None, None, None],
        })
        ex.to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)

        log = pd.DataFrame({
            "user_id": [1, 1, 2, 3],
            "exercise": ["ar1", "ge1", "bi1", "ar1"],
            "problem_type": [0, 0, 0, 0],
            "correct": ["true", "false", "true", "true"],
            "time_done": [100, 200, 300, 400],
        })
        log_path = raw_dir / "junyi_ProblemLog_original.csv"
        log.to_csv(log_path, index=False)

        out_path = tmp_path / "responses.parquet"
        stats = extract(log_path=log_path, out_path=out_path, raw_dir=raw_dir, chunk_size=2)

        assert stats["lignes_lues"] == 4
        assert stats["lignes_retenues"] == 3  # bi1 exclu (area biology)

        table = pq.read_table(out_path).to_pandas()
        assert len(table) == 3
        assert set(table["concept_id"]) == {"arithmetic", "geometry"}
        assert table["correct"].dtype == bool

    def test_chunking_ne_perd_ni_ne_duplique_de_lignes(self, tmp_path):
        """chunk_size plus petit que le nombre de lignes -- verifie que le
        decoupage en plusieurs chunks ne perd ni ne duplique de lignes."""
        raw_dir = tmp_path
        ex = pd.DataFrame({
            "name": ["ar1"], "area": ["arithmetic"],
            "topic": ["addition-subtraction"], "prerequisites": [None],
        })
        ex.to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)

        n = 27  # pas un multiple du chunk_size, expres
        log = pd.DataFrame({
            "user_id": range(n), "exercise": ["ar1"] * n,
            "correct": ["true"] * n, "time_done": range(n),
        })
        log_path = raw_dir / "junyi_ProblemLog_original.csv"
        log.to_csv(log_path, index=False)

        out_path = tmp_path / "responses.parquet"
        stats = extract(log_path=log_path, out_path=out_path, raw_dir=raw_dir, chunk_size=10)

        assert stats["lignes_retenues"] == n
        table = pq.read_table(out_path).to_pandas()
        assert len(table) == n
        assert sorted(table["timestamp"].tolist()) == list(range(n))
