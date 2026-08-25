"""
Tests de d0_first_attempt_variant.py sur fixtures synthetiques -- jamais le
fichier reel de 26M lignes.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from d0_first_attempt_variant import build_first_attempt_variant

DOMAIN_PATH = Path(__file__).resolve().parent.parent / "domain.yaml"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestBuildFirstAttemptVariant:

    def test_ne_garde_que_problem_number_1(self, tmp_path):
        raw_dir = tmp_path
        ex = pd.DataFrame({
            "name": ["ar1"], "area": ["arithmetic"],
            "topic": ["addition-subtraction"], "prerequisites": [None],
        })
        ex.to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)

        log = pd.DataFrame({
            "user_id": [1, 1, 1, 2],
            "exercise": ["ar1", "ar1", "ar1", "ar1"],
            "problem_number": [1, 2, 3, 1],
            "correct": ["false", "false", "true", "true"],
            "time_done": [100, 200, 300, 400],
        })
        log_path = raw_dir / "junyi_ProblemLog_original.csv"
        log.to_csv(log_path, index=False)

        out_path = tmp_path / "responses_first_attempt.parquet"
        stats = build_first_attempt_variant(log_path=log_path, out_path=out_path,
                                            raw_dir=raw_dir, chunk_size=2)

        assert stats["lignes_brutes_lues"] == 4
        assert stats["lignes_premiere_tentative_retenues"] == 2  # etudiant 1 pn=1, etudiant 2 pn=1

        table = pq.read_table(out_path).to_pandas()
        assert len(table) == 2
        # etudiant 1 a echoue a sa PREMIERE tentative (pn=1), pas a sa 3e
        # (qui a fini par reussir apres pratique) -- c'est precisement ce
        # que la variante doit exclure.
        row_e1 = table[table["student_id"] == 1].iloc[0]
        assert row_e1["correct"] == False

    def test_exclut_les_exercices_hors_domaine(self, tmp_path):
        raw_dir = tmp_path
        ex = pd.DataFrame({
            "name": ["ar1", "bi1"], "area": ["arithmetic", "biology"],
            "topic": ["addition-subtraction", "x"], "prerequisites": [None, None],
        })
        ex.to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)

        log = pd.DataFrame({
            "user_id": [1, 1], "exercise": ["ar1", "bi1"],
            "problem_number": [1, 1], "correct": ["true", "true"],
            "time_done": [100, 200],
        })
        log_path = raw_dir / "junyi_ProblemLog_original.csv"
        log.to_csv(log_path, index=False)

        out_path = tmp_path / "responses_first_attempt.parquet"
        stats = build_first_attempt_variant(log_path=log_path, out_path=out_path,
                                            raw_dir=raw_dir, chunk_size=10)
        assert stats["lignes_premiere_tentative_retenues"] == 1

    def test_ne_touche_jamais_domain_yaml(self, tmp_path):
        """E2 (docs/revue_d0_tour2.md) : la promesse de ne jamais ecrire
        dans domain.yaml ne tenait jusqu'ici qu'a l'absence d'appel a
        write_calibrated_domain dans le script -- verrouille ici par un
        hash avant/apres execution."""
        raw_dir = tmp_path
        ex = pd.DataFrame({
            "name": ["ar1"], "area": ["arithmetic"],
            "topic": ["addition-subtraction"], "prerequisites": [None],
        })
        ex.to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)
        log = pd.DataFrame({
            "user_id": [1], "exercise": ["ar1"], "problem_number": [1],
            "correct": ["true"], "time_done": [100],
        })
        log_path = raw_dir / "junyi_ProblemLog_original.csv"
        log.to_csv(log_path, index=False)

        assert DOMAIN_PATH.exists(), "domain.yaml de production introuvable"
        hash_avant = _hash(DOMAIN_PATH)

        build_first_attempt_variant(log_path=log_path, out_path=tmp_path / "out.parquet",
                                    raw_dir=raw_dir, chunk_size=10)

        assert _hash(DOMAIN_PATH) == hash_avant
