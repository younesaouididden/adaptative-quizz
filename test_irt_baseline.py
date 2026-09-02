"""Tests de irt_baseline.py, sur fixtures synthetiques (jamais domains/piste_b.yaml)."""

import numpy as np
import pytest

from irt_baseline import (
    build_irt_items,
    eap_theta,
    fisher_info_3pl,
    p_correct_3pl,
    select_next_irt,
    simulate_irt,
)
from kst_engine import Concept, Domain, Question


def _tiny_domain():
    concepts = [Concept("x"), Concept("y")]
    questions = [
        Question("x_1", "x", slip=0.10, guess=0.25),
        Question("x_2", "x", slip=0.10, guess=0.25),
        Question("y_1", "y", slip=0.10, guess=0.25),
        Question("y_2", "y", slip=0.10, guess=0.25),
    ]
    return Domain(concepts=concepts, prereqs=[], questions=questions)


def _tiny_meta():
    return [
        {"difficulty": 1}, {"difficulty": 3},
        {"difficulty": 1}, {"difficulty": 3},
    ]


def test_p_correct_3pl_equals_c_plus_half_1_minus_c_at_difficulty():
    # theta == b : le terme logistique vaut 1/2 (chapitre standard IRT)
    p = p_correct_3pl(theta=0.5, a=1.0, b=0.5, c=0.25)
    assert p == pytest.approx(0.25 + 0.5 * 0.75)


def test_p_correct_3pl_is_bounded_by_guessing_and_one():
    for theta in (-4.0, -1.0, 0.0, 1.0, 4.0):
        p = p_correct_3pl(theta, a=1.0, b=0.0, c=0.25)
        assert 0.25 <= p <= 1.0


def test_fisher_info_is_maximal_near_item_difficulty():
    # information de Fisher 3PL : maximale pres de theta=b (a discrimination
    # et guessing fixes), decroit en s'en eloignant
    b = 0.0
    info_at_b = fisher_info_3pl(theta=b, a=1.0, b=b, c=0.25)
    info_far = fisher_info_3pl(theta=b + 3.0, a=1.0, b=b, c=0.25)
    assert info_at_b > info_far


def test_eap_theta_no_responses_returns_prior_mean():
    theta_hat, se = eap_theta([])
    assert theta_hat == pytest.approx(0.0, abs=1e-6)
    assert se > 0.5  # ecart-type proche du prior N(0,1)


def test_eap_theta_recovers_known_theta():
    # test de recuperation de parametre (meme logique que le test EM du
    # projet, cf. data/test_calibrate_em.py) : on genere des reponses a
    # partir d'un theta connu et on verifie que l'EAP le retrouve -- en
    # moyenne sur plusieurs etudiants simules, pas sur un seul tirage (l'EAP
    # a un biais de retrecissement vers le prior a effectif fini, un seul
    # tirage bruite peut s'en ecarter meme si l'estimateur est correct).
    true_theta = 1.2
    items = [{"a": 1.0, "b": b, "c": 0.2} for b in np.linspace(-3, 3, 150)]

    estimates = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        responses = []
        for it in items:
            p = p_correct_3pl(true_theta, it["a"], it["b"], it["c"])
            correct = bool(rng.random() < p)
            responses.append((it["a"], it["b"], it["c"], correct))
        theta_hat, _ = eap_theta(responses)
        estimates.append(theta_hat)

    assert np.mean(estimates) == pytest.approx(true_theta, abs=0.2)


def test_select_next_irt_prefers_item_matched_to_current_theta():
    items = [{"a": 1.0, "b": -3.0, "c": 0.25},
            {"a": 1.0, "b": 0.0, "c": 0.25},
            {"a": 1.0, "b": 3.0, "c": 0.25}]
    q, _ = select_next_irt(theta_hat=0.0, items=items, asked=set())
    assert q == 1


def test_select_next_irt_skips_asked_items():
    items = [{"a": 1.0, "b": 0.0, "c": 0.25},
            {"a": 1.0, "b": 3.0, "c": 0.25}]
    q, _ = select_next_irt(theta_hat=0.0, items=items, asked={0})
    assert q == 1


def test_build_irt_items_maps_difficulty_to_b_and_guess_to_c():
    domain = _tiny_domain()
    items = build_irt_items(domain, _tiny_meta())
    assert items[0]["b"] == -1.0  # difficulte 1 -> b = 1-2
    assert items[1]["b"] == 1.0   # difficulte 3 -> b = 3-2
    assert items[0]["c"] == pytest.approx(0.25)


def test_simulate_irt_returns_consistent_shape():
    domain = _tiny_domain()
    meta = _tiny_meta()
    z_true = frozenset({"x"})

    r = simulate_irt(domain, meta, z_true, seed=0)

    assert r["n_questions"] > 0
    assert isinstance(r["z_hat"], frozenset)
    assert r["z_hat"] <= {"x", "y"}
    assert isinstance(r["correct_diagnosis"], bool)
    assert len(r["se_trace"]) == r["n_questions"] + 1


def test_simulate_irt_stops_at_max_questions():
    domain = _tiny_domain()
    meta = _tiny_meta()
    r = simulate_irt(domain, meta, frozenset({"x"}), seed=0, max_questions=2)
    assert r["n_questions"] <= 2
