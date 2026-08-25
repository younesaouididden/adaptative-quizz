"""
Tests de d0_control_random_sample.py sur fixtures synthetiques -- jamais
les fichiers reels.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from d0_control_random_sample import build_control_statistics


@pytest.fixture
def domain_file(tmp_path):
    data = {
        "schema_version": 1, "domain": "test",
        "concepts": [{"id": "a"}, {"id": "b"}],
        "prerequisites": [["a", "b"]],
    }
    path = tmp_path / "domain.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


class TestBuildControlStatistics:

    def test_K_egal_au_total_reproduit_exactement_le_pool(self, tmp_path, domain_file):
        """Si la variante contient AUTANT d'observations que le jeu complet
        pour une paire (K == n_total), il n'y a rien a tirer au hasard : le
        controle doit retomber exactement sur le pool complet."""
        full = pd.DataFrame({
            "student_id": [1, 1, 1], "concept_id": ["a", "a", "a"],
            "correct": [True, True, False],
        })
        variant = pd.DataFrame({
            "student_id": [1, 1, 1], "concept_id": ["a", "a", "a"],
            "correct": [True, False, True],  # valeurs sans importance, seul le compte K compte
        })
        full_path = tmp_path / "responses.parquet"
        variant_path = tmp_path / "responses_first_attempt.parquet"
        full.to_parquet(full_path)
        variant.to_parquet(variant_path)

        n_correct_control, K, concepts, Z = build_control_statistics(
            responses_path=full_path, variant_path=variant_path,
            domain_path=domain_file, seed=0)

        assert K[0, concepts.index("a")] == 3
        assert n_correct_control[0, concepts.index("a")] == 2  # = n_correct du pool complet

    def test_K_zero_donne_zero_correct(self, tmp_path, domain_file):
        # la variante (premiere tentative) est toujours un SOUS-ENSEMBLE du
        # pool complet -- ici l'etudiant n'a jamais touche "a" du tout, ni
        # dans le pool complet ni dans la variante (K=0 pour "a" est donc
        # coherent, pas une violation de l'invariant sous-ensemble).
        full = pd.DataFrame({"student_id": [1], "concept_id": ["b"], "correct": [True]})
        variant = pd.DataFrame({"student_id": [1], "concept_id": ["b"], "correct": [True]})
        full_path = tmp_path / "responses.parquet"
        variant_path = tmp_path / "responses_first_attempt.parquet"
        full.to_parquet(full_path)
        variant.to_parquet(variant_path)

        n_correct_control, K, concepts, Z = build_control_statistics(
            responses_path=full_path, variant_path=variant_path,
            domain_path=domain_file, seed=0)
        assert K[0, concepts.index("a")] == 0
        assert n_correct_control[0, concepts.index("a")] == 0

    def test_moyenne_hypergeometrique_conforme_sur_grand_echantillon(self, tmp_path, domain_file):
        """Propriete statistique : sur beaucoup de tirages, la moyenne du
        nombre de corrects tire doit converger vers K * (taux de reussite
        du pool complet) -- moyenne d'une hypergeometrique."""
        n_students = 2000
        rng = np.random.default_rng(0)
        student_ids = np.repeat(np.arange(n_students), 10)
        full = pd.DataFrame({
            "student_id": student_ids, "concept_id": "a",
            "correct": rng.random(len(student_ids)) < 0.7,  # 70% de reussite dans le pool
        })
        # variante : K=3 tirages par etudiant (sous-ensemble arbitraire)
        variant = pd.DataFrame({
            "student_id": np.repeat(np.arange(n_students), 3),
            "concept_id": "a", "correct": True,
        })
        full_path = tmp_path / "responses.parquet"
        variant_path = tmp_path / "responses_first_attempt.parquet"
        full.to_parquet(full_path)
        variant.to_parquet(variant_path)

        n_correct_control, K, concepts, Z = build_control_statistics(
            responses_path=full_path, variant_path=variant_path,
            domain_path=domain_file, seed=42)

        taux_moyen = n_correct_control[:, concepts.index("a")].sum() / K[:, concepts.index("a")].sum()
        assert taux_moyen == pytest.approx(0.7, abs=0.03)

    def test_leve_si_variante_depasse_le_pool_complet(self, tmp_path, domain_file):
        full = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "correct": [True]})
        variant = pd.DataFrame({
            "student_id": [1, 1], "concept_id": ["a", "a"], "correct": [True, True],
        })
        full_path = tmp_path / "responses.parquet"
        variant_path = tmp_path / "responses_first_attempt.parquet"
        full.to_parquet(full_path)
        variant.to_parquet(variant_path)

        with pytest.raises(ValueError, match="plus d'observations"):
            build_control_statistics(responses_path=full_path, variant_path=variant_path,
                                     domain_path=domain_file, seed=0)
