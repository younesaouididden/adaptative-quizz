import numpy as np
import pytest

from d0_full_comparison import concept_marginals_from_prior
from kst_engine import build_knowledge_space


def test_marginale_coherente_avec_prior_uniforme():
    Z = build_knowledge_space(["a", "b"], [("a", "b")])
    prior = np.full(len(Z), 1 / len(Z))
    marg = concept_marginals_from_prior(prior, Z, ["a", "b"])
    n_avec_a = sum(1 for z in Z if "a" in z)
    assert marg["a"] == pytest.approx(n_avec_a / len(Z))


def test_masse_concentree_sur_etat_vide_donne_marginales_nulles():
    Z = build_knowledge_space(["a", "b"], [("a", "b")])
    prior = np.zeros(len(Z))
    prior[Z.index(frozenset())] = 1.0
    marg = concept_marginals_from_prior(prior, Z, ["a", "b"])
    assert marg["a"] == pytest.approx(0.0)
    assert marg["b"] == pytest.approx(0.0)
