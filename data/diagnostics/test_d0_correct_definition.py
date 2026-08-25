"""
Tests du diagnostic D0 sur fixtures synthetiques -- jamais le fichier reel
de 26M lignes.
"""

import pandas as pd

from d0_correct_definition import (
    accumulate_correctness_and_review,
    accumulate_problem_number_histogram,
    check_pn_counts_non_increasing,
    spot_check_sequential,
)


def _write_log(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["user_id", "exercise", "problem_number",
                                     "hint_used", "count_hints"])
    path = tmp_path / "junyi_ProblemLog_original.csv"
    df.to_csv(path, index=False)
    return path


def _write_full_log(tmp_path, rows):
    """Comme _write_log, mais avec les colonnes 'correct' et 'review_mode'
    necessaires a accumulate_correctness_and_review (R1/R4/R5)."""
    df = pd.DataFrame(rows, columns=["problem_number", "correct", "review_mode"])
    path = tmp_path / "junyi_ProblemLog_original.csv"
    df.to_csv(path, index=False)
    return path


class TestAccumulateProblemNumberHistogram:

    def test_histogramme_correct_sur_petit_exemple(self, tmp_path):
        # etudiant 1 x exercice a : 3 tentatives (pn 1,2,3)
        # etudiant 2 x exercice a : 1 tentative (pn 1)
        rows = [
            (1, "a", 1, "false", 0),
            (1, "a", 2, "false", 0),
            (1, "a", 3, "false", 0),
            (2, "a", 1, "false", 0),
        ]
        path = _write_log(tmp_path, rows)
        hist = accumulate_problem_number_histogram(log_path=path, chunk_size=2)

        assert hist["n_rows"] == 4
        assert hist["pn_counts"][1] == 2   # 2 paires ont une ligne a pn=1
        assert hist["pn_counts"][2] == 1
        assert hist["pn_counts"][3] == 1
        assert hist["n_hint"] == 0

    def test_compte_les_indices(self, tmp_path):
        rows = [
            (1, "a", 1, "true", 1),
            (1, "a", 2, "false", 0),
        ]
        path = _write_log(tmp_path, rows)
        hist = accumulate_problem_number_histogram(log_path=path, chunk_size=10)
        assert hist["n_hint"] == 1

    def test_chunking_ne_change_pas_le_resultat(self, tmp_path):
        rows = [(1, "a", i, "false", 0) for i in range(1, 8)] + [(2, "b", 1, "false", 0)]
        path = _write_log(tmp_path, rows)
        hist_1_chunk = accumulate_problem_number_histogram(log_path=path, chunk_size=100)
        hist_small_chunks = accumulate_problem_number_histogram(log_path=path, chunk_size=3)
        assert hist_1_chunk["n_rows"] == hist_small_chunks["n_rows"] == 8
        assert hist_1_chunk["pn_counts"].equals(hist_small_chunks["pn_counts"])
        assert hist_1_chunk["pn_counts"][1] == 2  # 2 paires distinctes


class TestSpotCheckSequential:

    def test_detecte_sequence_correcte(self, tmp_path):
        rows = [(1, "a", 1, "f", 0), (1, "a", 2, "f", 0), (1, "a", 3, "f", 0)]
        path = _write_log(tmp_path, rows)
        result = spot_check_sequential(log_path=path, chunk_size=10)
        assert result["n_paires_testees"] == 1
        assert result["n_sequentielles_1_a_N"] == 1

    def test_detecte_sequence_avec_trou(self, tmp_path):
        # trou : 1, 3 (pas de 2) -- ne doit PAS compter comme sequentielle
        rows = [(1, "a", 1, "f", 0), (1, "a", 3, "f", 0)]
        path = _write_log(tmp_path, rows)
        result = spot_check_sequential(log_path=path, chunk_size=10)
        assert result["n_paires_testees"] == 1
        assert result["n_sequentielles_1_a_N"] == 0

    def test_reconstruit_historique_disperse_sur_plusieurs_chunks(self, tmp_path):
        """Les lignes d'une meme paire peuvent etre dispersees loin les
        unes des autres dans le fichier (fichier ordonne dans le temps) --
        le sondage corrige doit les recoller malgre un chunk_size petit."""
        rows = ([(1, "a", 1, "f", 0)]
               + [(90 + i, "z", 1, "f", 0) for i in range(5)]   # bruit, paires distinctes
               + [(1, "a", 2, "f", 0)])
        path = _write_log(tmp_path, rows)
        result = spot_check_sequential(log_path=path, candidate_scan_rows=100, chunk_size=2)
        assert result["n_paires_testees"] == 1
        assert result["n_sequentielles_1_a_N"] == 1


# ---------------------------------------------------------------------------
# R1 / R4 / R5 (docs/revue_d0.md)
# ---------------------------------------------------------------------------

class TestAccumulateCorrectnessAndReview:

    def test_taux_de_reussite_par_problem_number(self, tmp_path):
        # problem_number=1 : 1 correct / 2 lignes (50%) ; pn=2 : 2/2 (100%)
        rows = [
            (1, "true", "false"), (1, "false", "false"),
            (2, "true", "false"), (2, "true", "false"),
        ]
        path = _write_full_log(tmp_path, rows)
        result = accumulate_correctness_and_review(log_path=path, chunk_size=2)
        assert result["pn_correct"][1] == 1
        assert result["pn_counts"][1] == 2
        assert result["pn_correct"][2] == 2
        assert result["pn_counts"][2] == 2

    def test_compte_review_mode(self, tmp_path):
        rows = [(1, "true", "true"), (1, "true", "false"), (2, "true", "false")]
        path = _write_full_log(tmp_path, rows)
        result = accumulate_correctness_and_review(log_path=path, chunk_size=10)
        assert result["n_review"] == 1
        assert result["n_rows"] == 3


class TestCheckPnCountsNonIncreasing:

    def test_ok_si_decroissant(self):
        r = check_pn_counts_non_increasing(pd.Series({1: 100, 2: 50, 3: 10}))
        assert r["ok"]
        assert r["n_violations"] == 0

    def test_rapporte_la_vraie_valeur_de_problem_number_en_cas_de_hausse(self):
        """Regression : la premiere version rapportait une POSITION dans la
        serie triee (sans rapport avec problem_number des que l'index ne
        commence pas a 0 ou saute des valeurs), pas la vraie valeur."""
        r = check_pn_counts_non_increasing(pd.Series({1: 10, 5: 50, 9: 5}))
        assert not r["ok"]
        assert r["n_violations"] == 1
        assert r["premiere_violation_problem_number"] == 5   # pas "1" (la position)
        assert r["premiere_violation_hausse"] == 40
        assert r["n_paires_a_la_premiere_violation"] == 50

    def test_plateau_accepte(self):
        """Une egalite (pas de hausse stricte) ne doit pas etre rejetee --
        seule une VRAIE hausse signe une violation de la numerotation."""
        r = check_pn_counts_non_increasing(pd.Series({1: 10, 2: 10, 3: 5}))
        assert r["ok"]
