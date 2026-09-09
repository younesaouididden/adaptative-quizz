"""Suite pytest de d2_1_signature_temporelle.py -- fixtures synthetiques
fabriquees a la main, jamais responses.parquet reel (25,9M lignes)."""

import numpy as np
import pandas as pd
import pytest

from d2_1_signature_temporelle import (
    build_pairs_table,
    compute_midpoints,
    merge_half_stats,
    merge_pair_stats,
    partial_half_stats,
    partial_pair_stats,
    permutation_null,
    verdict,
)


class TestPartialPairStats:

    def test_agrege_par_couple(self):
        chunk = pd.DataFrame({
            "student_id": [1, 1, 1, 2],
            "concept_id": ["a", "a", "b", "a"],
            "correct": [True, False, True, True],
            "timestamp": [100, 200, 150, 300],
        })
        out = partial_pair_stats(chunk).set_index(["student_id", "concept_id"])
        assert out.loc[(1, "a")].tolist() == [1, 2, 100, 200]  # n_correct, n_total, min, max
        assert out.loc[(1, "b")].tolist() == [1, 1, 150, 150]
        assert out.loc[(2, "a")].tolist() == [1, 1, 300, 300]


class TestMergePairStats:

    def test_combine_deux_chunks_du_meme_couple(self):
        acc = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "n_correct": [3],
                            "n_total": [5], "min_ts": [100], "max_ts": [400]})
        partial = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "n_correct": [2],
                                "n_total": [4], "min_ts": [50], "max_ts": [600]})
        out = merge_pair_stats(acc, partial).set_index(["student_id", "concept_id"])
        row = out.loc[(1, "a")]
        assert row["n_correct"] == 5
        assert row["n_total"] == 9
        assert row["min_ts"] == 50
        assert row["max_ts"] == 600

    def test_acc_none_retourne_le_partiel(self):
        partial = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "n_correct": [1],
                                "n_total": [1], "min_ts": [1], "max_ts": [1]})
        out = merge_pair_stats(None, partial)
        pd.testing.assert_frame_equal(out, partial)


class TestComputeMidpoints:

    def test_filtre_sur_seuil_minimum(self):
        pair_stats = pd.DataFrame({
            "student_id": [1, 2], "concept_id": ["a", "a"],
            "n_correct": [10, 5], "n_total": [20, 10],
            "min_ts": [0, 0], "max_ts": [1000, 1000],
        })
        out = compute_midpoints(pair_stats, min_n=20)
        assert len(out) == 1
        assert out.iloc[0]["student_id"] == 1

    def test_point_milieu_est_le_centre_temporel(self):
        pair_stats = pd.DataFrame({
            "student_id": [1], "concept_id": ["a"], "n_correct": [10],
            "n_total": [20], "min_ts": [100], "max_ts": [300],
        })
        out = compute_midpoints(pair_stats, min_n=20)
        assert out.iloc[0]["midpoint"] == 200

    def test_exclut_couple_sans_etalement_temporel(self):
        """min_ts == max_ts (toutes les reponses au meme instant) : aucun
        decoupage possible, doit etre exclu plutot que de produire une
        moitie vide."""
        pair_stats = pd.DataFrame({
            "student_id": [1], "concept_id": ["a"], "n_correct": [10],
            "n_total": [20], "min_ts": [500], "max_ts": [500],
        })
        out = compute_midpoints(pair_stats, min_n=20)
        assert len(out) == 0


class TestPartialHalfStats:

    def test_decoupe_par_le_temps_pas_par_le_compte(self):
        """3 reponses avant le milieu, 1 apres -- pas un decoupage 2/2 par
        nombre d'essais, exactement la distinction que D2.1 doit respecter."""
        chunk = pd.DataFrame({
            "student_id": [1, 1, 1, 1],
            "concept_id": ["a", "a", "a", "a"],
            "correct": [True, False, True, True],
            "timestamp": [10, 20, 30, 90],
        })
        midpoints = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "midpoint": [50.0]})
        out = partial_half_stats(chunk, midpoints).set_index(["student_id", "concept_id", "half"])
        assert out.loc[(1, "a", "first")].tolist() == [2, 3]   # n_correct, n_total
        assert out.loc[(1, "a", "second")].tolist() == [1, 1]

    def test_couple_non_retenu_absent_du_resultat(self):
        chunk = pd.DataFrame({
            "student_id": [1, 2], "concept_id": ["a", "a"],
            "correct": [True, True], "timestamp": [10, 10],
        })
        midpoints = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "midpoint": [5.0]})
        out = partial_half_stats(chunk, midpoints)
        assert set(out["student_id"]) == {1}

    def test_chunk_sans_couple_retenu_renvoie_vide(self):
        chunk = pd.DataFrame({"student_id": [9], "concept_id": ["z"],
                              "correct": [True], "timestamp": [1]})
        midpoints = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "midpoint": [5.0]})
        out = partial_half_stats(chunk, midpoints)
        assert len(out) == 0


class TestMergeHalfStats:

    def test_combine_deux_chunks(self):
        acc = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "half": ["first"],
                            "n_correct": [2], "n_total": [3]})
        partial = pd.DataFrame({"student_id": [1], "concept_id": ["a"], "half": ["first"],
                                "n_correct": [1], "n_total": [2]})
        out = merge_half_stats(acc, partial).set_index(["student_id", "concept_id", "half"])
        row = out.loc[(1, "a", "first")]
        assert row["n_correct"] == 3
        assert row["n_total"] == 5


class TestBuildPairsTable:

    def test_calcule_delta_et_filtre_la_bande(self):
        retained = pd.DataFrame({
            "student_id": [1, 2], "concept_id": ["a", "a"],
            "n_correct": [6, 19], "n_total": [10, 20],   # rate_global 0.6 et 0.95
            "min_ts": [0, 0], "max_ts": [100, 100], "midpoint": [50, 50],
        })
        half_stats = pd.DataFrame({
            "student_id": [1, 1, 2, 2], "concept_id": ["a", "a", "a", "a"],
            "half": ["first", "second", "first", "second"],
            "n_correct": [2, 4, 9, 10], "n_total": [5, 5, 10, 10],
        })
        out = build_pairs_table(retained, half_stats, rate_band=(0.4, 0.8))
        assert len(out) == 1  # etudiant 2 (rate_global=0.95) hors bande
        row = out.iloc[0]
        assert row["student_id"] == 1
        assert row["rate1"] == pytest.approx(0.4)
        assert row["rate2"] == pytest.approx(0.8)
        assert row["delta"] == pytest.approx(0.4)

    def test_exclut_moitie_vide(self):
        retained = pd.DataFrame({
            "student_id": [1], "concept_id": ["a"], "n_correct": [5], "n_total": [10],
            "min_ts": [0], "max_ts": [100], "midpoint": [50],
        })
        half_stats = pd.DataFrame({
            "student_id": [1], "concept_id": ["a"], "half": ["first"],
            "n_correct": [5], "n_total": [10],
        })
        out = build_pairs_table(retained, half_stats, rate_band=(0.4, 0.8))
        assert len(out) == 0


class TestPermutationNull:

    def test_moyenne_nulle_proche_de_zero(self):
        """Sous permutation, E[delta] = 0 par construction (hypergeometrique
        centree). Verifie sur des couples avec suffisamment de couples/
        replications pour que le bruit d'echantillonnage reste petit."""
        rng_data = np.random.default_rng(0)
        n = 300
        pairs = pd.DataFrame({
            "n_correct": rng_data.integers(10, 30, n),
            "n_total": 40,
            "n1": 20,
            "n2": 20,
        })
        rng = np.random.default_rng(1)
        null_agg = permutation_null(pairs, n_reps=2000, rng=rng)
        assert null_agg.mean() == pytest.approx(0.0, abs=0.01)

    def test_taille_de_sortie(self):
        pairs = pd.DataFrame({"n_correct": [5, 8], "n_total": [10, 10],
                              "n1": [5, 5], "n2": [5, 5]})
        rng = np.random.default_rng(0)
        out = permutation_null(pairs, n_reps=37, rng=rng)
        assert out.shape == (37,)


class TestVerdict:

    def test_confirme_si_hausse_substantielle_et_significative(self):
        v = verdict(mean_delta=0.10, p_value=0.001)
        assert v.startswith("D2 CONFIRMEE")

    def test_rejete_si_hausse_trop_faible(self):
        v = verdict(mean_delta=0.01, p_value=0.0001)
        assert v.startswith("D2 REJETEE")

    def test_rejete_si_ne_survit_pas_au_controle(self):
        v = verdict(mean_delta=0.10, p_value=0.5)
        assert v.startswith("D2 REJETEE")
