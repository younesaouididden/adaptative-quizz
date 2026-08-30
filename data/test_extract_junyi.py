"""
Suite pytest de extract_junyi.py -- fixtures synthetiques, jamais le fichier
Junyi reel de 837 lignes (rapide, independant de data/raw/).
"""

from collections import Counter

import pandas as pd
import pytest

from extract_junyi import (
    ALPHA,
    EXCLUDED_AREAS,
    build_domain,
    build_exercise_edges,
    concept_of,
    aggregate_to_concepts,
    resolve_conflicts,
)
from kst_engine import build_knowledge_space


# ---------------------------------------------------------------------------
# build_exercise_edges -- regression sur le bug de parsing des listes
# ---------------------------------------------------------------------------

class TestBuildExerciseEdges:

    def test_prerequisite_unique(self):
        ex = pd.DataFrame({"name": ["b"], "prerequisites": ["a"]})
        assert build_exercise_edges(ex) == [("a", "b")]

    def test_prerequisites_multiples_separees_par_virgule(self):
        """Regression : un premier passage traitait 'a,b' comme UN SEUL nom
        d'exercice inexistant plutot que deux prerequis distincts -- ce qui
        avait fait disparaitre les vrais prerequis de 'calculus' et l'avait
        rendu isole a tort lors de l'exploration initiale."""
        ex = pd.DataFrame({"name": ["c"], "prerequisites": ["a, b"]})
        assert set(build_exercise_edges(ex)) == {("a", "c"), ("b", "c")}

    def test_prerequisite_absent_ignore(self):
        ex = pd.DataFrame({"name": ["a", "b"], "prerequisites": [None, "a"]})
        assert build_exercise_edges(ex) == [("a", "b")]


# ---------------------------------------------------------------------------
# concept_of -- V1 simplifiee : concept = area brute normalisee
# ---------------------------------------------------------------------------

class TestConceptOf:

    def test_area_normalisee_underscore(self):
        assert concept_of("probability-statistics") == "probability_statistics"

    def test_area_sans_tiret_inchangee(self):
        assert concept_of("geometry") == "geometry"

    def test_area_exclue_renvoie_none(self):
        for area in EXCLUDED_AREAS:
            assert concept_of(area) is None

    def test_area_nan_renvoie_none(self):
        assert concept_of(float("nan")) is None


# ---------------------------------------------------------------------------
# aggregate_to_concepts
# ---------------------------------------------------------------------------

class TestAggregateToConcepts:

    def test_edges_intra_concept_exclues(self):
        ex = pd.DataFrame({
            "name": ["a1", "a2"],
            "area": ["arithmetic", "arithmetic"],
        })
        _, dir_counts = aggregate_to_concepts(ex, [("a1", "a2")])
        assert dir_counts == Counter()

    def test_edges_inter_concept_comptees(self):
        ex = pd.DataFrame({
            "name": ["a1", "g1"],
            "area": ["arithmetic", "geometry"],
        })
        name_to_concept, dir_counts = aggregate_to_concepts(ex, [("a1", "g1")])
        assert name_to_concept == {"a1": "arithmetic", "g1": "geometry"}
        assert dir_counts[("arithmetic", "geometry")] == 1

    def test_exercice_exclu_absent_du_mapping(self):
        ex = pd.DataFrame({"name": ["b1"], "area": ["biology"]})
        name_to_concept, dir_counts = aggregate_to_concepts(ex, [])
        assert name_to_concept == {}

    def test_calculus_et_logics_exclus(self):
        ex = pd.DataFrame({"name": ["c1", "l1"], "area": ["calculus", "logics"]})
        name_to_concept, _ = aggregate_to_concepts(ex, [])
        assert name_to_concept == {}


# ---------------------------------------------------------------------------
# resolve_conflicts -- coeur du test binomial
# ---------------------------------------------------------------------------

class TestResolveConflicts:

    def test_lien_unanime_garde_sans_test(self):
        kept, report = resolve_conflicts(Counter({("a", "b"): 5}))
        assert kept == [("a", "b")]
        assert report[0]["decision"] == "unanime"

    def test_conflit_tres_significatif_tranche_par_majorite(self):
        # 16 contre 1 : p << alpha, doit trancher vers la majorite
        kept, report = resolve_conflicts(Counter({("a", "b"): 16, ("b", "a"): 1}))
        assert kept == [("a", "b")]
        assert report[0]["decision"] == "significatif"

    def test_conflit_faible_rejete_comme_indetermine(self):
        # 2 contre 1 : ecart trivialement explicable par le hasard (p=1.0
        # pour un test binomial bilateral a n=3) -- doit rester indetermine,
        # pas de lien force.
        kept, report = resolve_conflicts(Counter({("a", "b"): 2, ("b", "a"): 1}))
        assert kept == []
        assert report[0]["decision"] == "indetermine (rejete)"

    def test_seuil_alpha_uniforme_pas_par_paire(self):
        """Le seuil par defaut du module doit etre le meme pour toutes les
        paires -- pas de logique per-pair qui ajusterait alpha apres coup."""
        assert ALPHA == 0.10

    def test_aucun_conflit_pas_de_paire_a_tort(self):
        kept, report = resolve_conflicts(Counter({("a", "b"): 3, ("c", "d"): 2}))
        assert set(kept) == {("a", "b"), ("c", "d")}
        assert all(r["decision"] == "unanime" for r in report)


# ---------------------------------------------------------------------------
# build_domain -- integration sur un mini-dataset synthetique
# ---------------------------------------------------------------------------

class TestBuildDomain:

    def test_domaine_synthetique_acyclique_et_connexe(self, tmp_path):
        """Petit dataset synthetique reproduisant la forme du vrai probleme
        (conflit faible a ignorer, lien fort a garder, areas exclues) --
        verifie tout le pipeline sans toucher au fichier Junyi reel."""
        ex = pd.DataFrame({
            "name":  ["ar1", "ar2", "ar3", "ge1", "bi1", "ca1"],
            "area":  ["arithmetic", "arithmetic", "arithmetic", "geometry", "biology", "calculus"],
            "prerequisites": [None, "ar1", "ar1", "ar1", None, None],
        })
        raw_dir = tmp_path
        ex.to_csv(raw_dir / "junyi_Exercise_table.csv", index=False)

        concepts, prereqs, report, name_to_concept = build_domain(raw_dir=raw_dir)

        assert set(concepts) == {"arithmetic", "geometry"}  # biology/calculus exclus
        assert ("arithmetic", "geometry") in prereqs

        Z = build_knowledge_space(concepts, prereqs)
        assert len(Z) <= 2 ** len(concepts)  # acyclique (sinon ValueError deja levee)

    def test_leve_si_lien_editorial_introduit_un_cycle(self, tmp_path, monkeypatch):
        """Si un lien editorial (MANUAL_EDGES) contredit les liens empiriques
        au point de creer un cycle, build_domain doit lever plutot que
        produire silencieusement un domaine incoherent."""
        import extract_junyi
        ex = pd.DataFrame({
            "name":  ["ar1", "ge1"],
            "area":  ["arithmetic", "geometry"],
            "prerequisites": [None, "ar1"],
        })
        ex.to_csv(tmp_path / "junyi_Exercise_table.csv", index=False)
        monkeypatch.setattr(extract_junyi, "MANUAL_EDGES",
                            [("geometry", "arithmetic")])
        with pytest.raises(ValueError, match="Cycle"):
            build_domain(raw_dir=tmp_path)
