"""
Suite pytest de calibrate_em.py -- fixtures synthetiques et un domaine
jouet genere avec des slip/guess CONNUS, pour verifier que l'EM les
retrouve (test standard de correction d'un EM : recuperation de parametres).
"""

import numpy as np
import pandas as pd
import pytest

from calibrate_em import (
    MIN_GAP,
    e_step,
    m_step,
    membership_matrix,
    run_em,
    sufficient_statistics,
)
from kst_engine import build_knowledge_space


# ---------------------------------------------------------------------------
# membership_matrix / sufficient_statistics
# ---------------------------------------------------------------------------

class TestMembershipMatrix:

    def test_indicatrice_correcte(self):
        Z = build_knowledge_space(["a", "b"], [("a", "b")])
        M = membership_matrix(Z, ["a", "b"])
        for z, row in zip(Z, M):
            for c_idx, c in enumerate(["a", "b"]):
                assert row[c_idx] == (1.0 if c in z else 0.0)


class TestSufficientStatistics:

    def test_agrege_correctement_par_etudiant_et_concept(self):
        responses = pd.DataFrame({
            "student_id": [1, 1, 1, 2],
            "concept_id": ["a", "a", "b", "a"],
            "correct": [True, False, True, True],
        })
        n_correct, n_total, students = sufficient_statistics(responses, ["a", "b"])
        assert students == [1, 2]
        # etudiant 1 : concept a -> 1 correct / 2 total ; concept b -> 1/1
        assert n_correct[0].tolist() == [1, 1]
        assert n_total[0].tolist() == [2, 1]
        # etudiant 2 : concept a -> 1/1 ; concept b -> jamais touche -> 0/0
        assert n_correct[1].tolist() == [1, 0]
        assert n_total[1].tolist() == [1, 0]


# ---------------------------------------------------------------------------
# e_step / m_step
# ---------------------------------------------------------------------------

class TestEStep:

    def test_cas_dor_monographie_via_e_step(self):
        """Meme cas de reference que kst_engine (0.500 -> 0.818), reformule
        comme un seul etudiant avec une seule reponse correcte sur un
        domaine a 1 concept."""
        Z = build_knowledge_space(["integrales"], [])
        M = membership_matrix(Z, ["integrales"])
        n_correct = np.array([[1.0]])
        n_total = np.array([[1.0]])
        prior = np.full(len(Z), 1 / len(Z))
        gamma, _ = e_step(n_correct, n_total, slip=np.array([0.10]),
                          guess=np.array([0.20]), prior=prior, M=M)
        z_avec_integrales = [i for i, z in enumerate(Z) if "integrales" in z][0]
        assert gamma[0, z_avec_integrales] == pytest.approx(0.818, abs=1e-3)

    def test_log_vraisemblance_augmente_avec_de_meilleurs_parametres(self):
        """Si les vrais parametres expliquent mieux les donnees que des
        parametres degrades, la log-vraisemblance doit etre plus haute."""
        Z = build_knowledge_space(["a"], [])
        M = membership_matrix(Z, ["a"])
        # etudiant qui maitrise "a" et repond juste 9 fois sur 10
        n_correct, n_total = np.array([[9.0]]), np.array([[10.0]])
        prior = np.full(len(Z), 0.5)
        _, ll_bons = e_step(n_correct, n_total, np.array([0.05]), np.array([0.20]), prior, M)
        _, ll_mauvais = e_step(n_correct, n_total, np.array([0.5]), np.array([0.5]), prior, M)
        assert ll_bons > ll_mauvais


class TestMStep:

    def test_recupere_slip_nul_si_toujours_correct_quand_maitrise(self):
        Z = build_knowledge_space(["a"], [])
        M = membership_matrix(Z, ["a"])
        z_maitrise = [i for i, z in enumerate(Z) if "a" in z][0]
        gamma = np.zeros((1, len(Z)))
        gamma[0, z_maitrise] = 1.0   # croyance certaine : etudiant maitrise "a"
        n_correct, n_total = np.array([[10.0]]), np.array([[10.0]])
        _, slip_new, _ = m_step(n_correct, n_total, gamma, M,
                                slip_prev=np.array([0.5]), guess_prev=np.array([0.5]))
        assert slip_new[0] == pytest.approx(0.0)

    def test_contrainte_projetee_si_violee(self):
        """Si le MLE non contraint donnerait slip+guess >= 1, la projection
        doit ramener sous 1-MIN_GAP en preservant le ratio slip/guess."""
        Z = build_knowledge_space(["a"], [])
        M = membership_matrix(Z, ["a"])
        gamma = np.array([[0.5, 0.5]]) if len(Z) == 2 else np.full((1, len(Z)), 1 / len(Z))
        # concu pour forcer slip et guess tous deux proches de 1 :
        # mastered : jamais correct (slip -> 1) ; unmastered : toujours correct (guess -> 1)
        n_correct = np.array([[0.0]])
        n_total = np.array([[1.0]])
        prior_new, slip_new, guess_new = m_step(
            n_correct, n_total, gamma, M,
            slip_prev=np.array([0.5]), guess_prev=np.array([0.5]))
        assert slip_new[0] + guess_new[0] <= 1 - MIN_GAP + 1e-9

    def test_concept_sans_signal_garde_valeur_precedente(self):
        Z = build_knowledge_space(["a"], [])
        M = membership_matrix(Z, ["a"])
        gamma = np.full((1, len(Z)), 1 / len(Z))
        n_correct, n_total = np.array([[0.0]]), np.array([[0.0]])  # jamais observe
        _, slip_new, guess_new = m_step(n_correct, n_total, gamma, M,
                                        slip_prev=np.array([0.13]), guess_prev=np.array([0.27]))
        assert slip_new[0] == pytest.approx(0.13)
        assert guess_new[0] == pytest.approx(0.27)


# ---------------------------------------------------------------------------
# run_em -- recuperation de parametres sur donnees synthetiques
# ---------------------------------------------------------------------------

class TestRunEM:

    def test_log_vraisemblance_monotone_croissante(self):
        """Propriete fondamentale de l'EM : chaque iteration ne peut QUE
        augmenter (ou stagner) la log-vraisemblance des donnees."""
        Z = build_knowledge_space(["a", "b"], [("a", "b")])
        rng = np.random.default_rng(0)
        n_students = 200
        n_correct = rng.integers(0, 10, size=(n_students, 2)).astype(float)
        n_total = n_correct + rng.integers(0, 10, size=(n_students, 2)).astype(float)
        result = run_em(n_correct, n_total, Z, ["a", "b"], rng, max_iter=30)
        trace = np.array(result.log_likelihood_trace)
        assert (np.diff(trace) >= -1e-6).all()  # tolerance flottante

    def test_recupere_des_parametres_synthetiques_connus(self):
        """Test standard de correction d'un EM : on genere des donnees a
        partir de slip/guess CONNUS, puis on verifie que l'EM les retrouve
        approximativement (tolerance large : l'EM ne recupere jamais les
        vrais parametres a la decimale pres sur un echantillon fini)."""
        concepts = ["a", "b"]
        Z = build_knowledge_space(concepts, [("a", "b")])
        true_slip = np.array([0.08, 0.15])
        true_guess = np.array([0.20, 0.25])
        M = membership_matrix(Z, concepts)

        rng = np.random.default_rng(42)
        n_students = 3000
        n_per_concept = 20
        # etats vrais tires uniformement dans Z
        z_true_idx = rng.integers(0, len(Z), size=n_students)
        true_P = M[z_true_idx] * (1 - true_slip) + (1 - M[z_true_idx]) * true_guess
        n_total = np.full((n_students, len(concepts)), float(n_per_concept))
        n_correct = rng.binomial(n_per_concept, true_P).astype(float)

        result = run_em(n_correct, n_total, Z, concepts, rng, max_iter=200, tol=1e-6)

        assert result.slip == pytest.approx(true_slip, abs=0.03)
        assert result.guess == pytest.approx(true_guess, abs=0.03)

    def test_deterministe_a_seed_fixe(self):
        Z = build_knowledge_space(["a"], [])
        n_correct = np.array([[5.0], [8.0]])
        n_total = np.array([[10.0], [10.0]])
        r1 = run_em(n_correct, n_total, Z, ["a"], np.random.default_rng(7), max_iter=20)
        r2 = run_em(n_correct, n_total, Z, ["a"], np.random.default_rng(7), max_iter=20)
        assert r1.slip == pytest.approx(r2.slip)
        assert r1.guess == pytest.approx(r2.guess)
