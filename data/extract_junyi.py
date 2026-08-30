"""
Tache A1 (ADDENDUM_BANQUE_QUESTIONS.md) : construit domain.yaml (concepts +
prerequis, chapitre 2 de la KST) a partir des metadonnees Junyi15
(junyi_Exercise_table.csv).

Ne construit PAS responses.parquet (Phase 3, script separe une fois ce
domaine valide -- la calibration EM en depend, donc autant valider le
graphe d'abord).

Pas de slip/guess/questions dans ce YAML : piste A calibre slip/guess par
EM sur les logs reels (tache A2), ce fichier n'a pas besoin des champs de
banque de questions de piste B.

SIMPLIFICATION V1 (2026-08-25) : la version precedente subdivisait
arithmetic/algebra par topic pour atteindre 9 concepts (cf. git history).
Le diagnostic docs/diagnostic_guess.md (D0/tour 2) a montre que la
granularite fine degradait l'identifiabilite de l'EM (trop peu
d'observations par (etudiant, concept) une fois subdivise). Avec la
deadline de stage qui approche, on revient a la granularite 'area' brute
(5 concepts, deja connexes -- cf. exploration A1) pour une V1 qui
fonctionne, quitte a affiner la granularite dans une iteration
ulterieure. calculus/logics restent exclus : trop peu d'exercices (10 et
5) pour peser sur l'identifiabilite meme a cette granularite.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kst_engine import build_knowledge_space  # reutilise, ne duplique pas

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT_PATH = Path(__file__).resolve().parent / "domain.yaml"

# Seuil ajuste depuis le seuil "theorique" 0.05 (test binomial bilateral)
# pour tenir compte des tres petits effectifs empiriques sur ce dataset.
# Decide AVANT de lancer l'analyse et applique uniformement a toutes les
# paires, pour ne pas biaiser le choix en fonction du resultat qu'on
# prefererait obtenir (discussion PROMPT_PISTE_A, 2026-08-25).
ALPHA = 0.10

# V1 simplifiee : granularite 'area' brute, pas de subdivision par topic.
# biology (composante isolee, aucun lien de prerequis vers le reste),
# calculus et logics (10 et 5 exercices -- trop peu pour peser sur
# l'identifiabilite, cf. docstring module) sont exclus.
EXCLUDED_AREAS = {"biology", "calculus", "logics"}

# Liens editoriaux (non derives des donnees) : aucun pour l'instant. Le
# lien calculus -> probability_statistics envisage a une granularite plus
# fine (PROMPT_PISTE_A, 2026-08-25) ne s'applique plus, calculus etant
# exclu de cette V1.
MANUAL_EDGES: list[tuple[str, str]] = []

CONCEPT_LABELS = {
    "arithmetic": "Arithmetique",
    "algebra": "Algebre",
    "geometry": "Geometrie",
    "analytic_geometry": "Geometrie analytique et trigonometrie",
    "probability_statistics": "Probabilites et statistiques",
}


def load_exercise_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "junyi_Exercise_table.csv")


def concept_of(area: str) -> str | None:
    """Concept = l'area Junyi elle-meme (granularite V1, voir docstring
    module), normalisee en style underscore. None si l'exercice doit etre
    exclu (area isolee, trop petite, ou manquante)."""
    if pd.isna(area) or area in EXCLUDED_AREAS:
        return None
    return area.replace("-", "_")


def build_exercise_edges(ex: pd.DataFrame) -> list[tuple[str, str]]:
    """Chapitre 2 (surmise) : (a, b) = a est prerequis de b.

    ATTENTION : la colonne 'prerequisites' contient une LISTE separee par
    virgules (189/742 lignes en ont plusieurs), pas une valeur unique --
    bug trouve et corrige lors de l'exploration (un premier passage qui
    traitait la liste entiere comme un seul nom d'exercice faisait
    disparaitre les vrais prerequis de 'calculus', le rendant isole a
    tort).
    """
    edges = []
    for _, row in ex.dropna(subset=["prerequisites"]).iterrows():
        child = row["name"]
        for parent in str(row["prerequisites"]).split(","):
            parent = parent.strip()
            if parent:
                edges.append((parent, child))
    return edges


def aggregate_to_concepts(ex: pd.DataFrame, edges: list[tuple[str, str]]
                          ) -> tuple[dict[str, str], Counter]:
    """Mappe chaque exercice a un concept, compte les liens diriges
    concept->concept (les liens intra-concept ne comptent pas : ils ne
    disent rien sur l'ordre entre CONCEPTS)."""
    name_to_area = dict(zip(ex["name"], ex["area"]))
    name_to_concept = {
        name: c for name in ex["name"]
        if (c := concept_of(name_to_area.get(name))) is not None
    }

    dir_counts = Counter()
    for p, c in edges:
        cp, cc = name_to_concept.get(p), name_to_concept.get(c)
        if cp is None or cc is None or cp == cc:
            continue
        dir_counts[(cp, cc)] += 1
    return name_to_concept, dir_counts


def resolve_conflicts(dir_counts: Counter, alpha: float = ALPHA
                      ) -> tuple[list[tuple[str, str]], list[dict]]:
    """Pour chaque paire de concepts reliee empiriquement, decide du sens du
    lien de prerequis (ou de son absence) par un test binomial bilateral :
    H0 = la direction observee est un tirage a 50/50 (bruit), H1 = un vrai
    biais directionnel existe. Seuil ALPHA, applique uniformement (voir
    commentaire en tete de fichier) pour eviter tout biais de selection.

    Retourne (aretes retenues, rapport d'arbitrage complet -- y compris les
    paires rejetees comme indeterminees) pour tracabilite dans le memoire.
    """
    pairs = {frozenset((a, b)) for a, b in dir_counts if a != b}
    kept: list[tuple[str, str]] = []
    report: list[dict] = []

    for pr in pairs:
        a, b = tuple(pr)
        ab = dir_counts.get((a, b), 0)   # votes pour "a prerequis de b"
        ba = dir_counts.get((b, a), 0)   # votes pour "b prerequis de a"
        n = ab + ba

        if ba == 0:                       # unanime, aucun test necessaire
            kept.append((a, b))
            report.append({"pair": f"{a} -> {b}", "votes": f"{ab} contre {ba}",
                           "decision": "unanime"})
            continue
        if ab == 0:
            kept.append((b, a))
            report.append({"pair": f"{b} -> {a}", "votes": f"{ba} contre {ab}",
                           "decision": "unanime"})
            continue

        edge = (a, b) if ab >= ba else (b, a)
        count_dom, count_rev = max(ab, ba), min(ab, ba)
        pval = binomtest(count_dom, n, 0.5, alternative="two-sided").pvalue
        if pval < alpha:
            kept.append(edge)
            report.append({"pair": f"{edge[0]} -> {edge[1]}",
                           "votes": f"{count_dom} contre {count_rev}",
                           "p_value": round(pval, 4), "decision": "significatif"})
        else:
            report.append({"pair": f"{a} <-> {b}", "votes": f"{ab} contre {ba}",
                           "p_value": round(pval, 4), "decision": "indetermine (rejete)"})

    return kept, report


def build_domain(raw_dir: Path = RAW_DIR, alpha: float = ALPHA
                 ) -> tuple[list[str], list[tuple[str, str]], list[dict], dict[str, str]]:
    """Pipeline complet : table -> aretes exercice -> agregation concept ->
    resolution de conflits -> ajout des liens editoriaux -> validation
    acyclique. Leve ValueError (via build_knowledge_space) si un cycle
    subsiste apres arbitrage."""
    ex = load_exercise_table(raw_dir)
    edges = build_exercise_edges(ex)
    name_to_concept, dir_counts = aggregate_to_concepts(ex, edges)
    kept, report = resolve_conflicts(dir_counts, alpha=alpha)

    concepts = sorted(set(name_to_concept.values()))
    concept_set = set(concepts)
    # les liens editoriaux ne s'appliquent que si les deux concepts existent
    # reellement dans CE dataset (ex. un sous-echantillon synthetique pour
    # les tests peut ne contenir ni 'calculus' ni 'probability_statistics').
    applicable_manual_edges = [(a, b) for a, b in MANUAL_EDGES
                              if a in concept_set and b in concept_set]
    prereqs = sorted(set(kept) | set(applicable_manual_edges))

    build_knowledge_space(concepts, prereqs)  # leve si cycle -- ne doit jamais arriver ici

    return concepts, prereqs, report, name_to_concept


def write_domain_yaml(concepts: list[str], prereqs: list[tuple[str, str]],
                      out_path: Path = OUT_PATH) -> None:
    data = {
        "schema_version": 1,
        "domain": "junyi_math_v1",
        "description": ("Domaine derive de Junyi15 (piste A), V1 simplifiee : "
                        "concepts = areas Junyi brutes (biology/calculus/logics "
                        "exclus), prerequis par test binomial (alpha="
                        f"{ALPHA}). slip/guess non presents : calibres par EM "
                        "en tache A2."),
        "concepts": [{"id": c, "label": CONCEPT_LABELS.get(c, c)} for c in concepts],
        "prerequisites": [[a, b] for a, b in prereqs],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    concepts, prereqs, report, name_to_concept = build_domain()
    Z = build_knowledge_space(concepts, prereqs)

    print(f"Concepts ({len(concepts)}) : {concepts}")
    print(f"2^n = {2**len(concepts)}   |Z| = {len(Z)}"
          f"   ({100*len(Z)/2**len(concepts):.1f}% du power set)")
    print()
    print(f"Prerequis retenus ({len(prereqs)}) :")
    for a, b in prereqs:
        tag = " [editorial]" if (a, b) in MANUAL_EDGES else ""
        print(f"  {a} -> {b}{tag}")
    print()
    print("Rapport d'arbitrage des conflits de direction :")
    for r in report:
        extra = f" (p={r['p_value']})" if "p_value" in r else ""
        print(f"  {r['pair']:55s} {r['votes']:14s} -> {r['decision']}{extra}")
    print()
    from collections import Counter as _C
    sizes = _C(name_to_concept.values())
    print("Exercices par concept :")
    for c in concepts:
        print(f"  {c:20s} : {sizes[c]}")
    print(f"  (total mappe : {sum(sizes.values())} / 837)")

    # Fourchette [30,500] de l'addendum pensee pour 8-14 concepts. La V1
    # simplifiee (5 concepts, cf. docstring module) descend mecaniquement en
    # dessous -- borne basse assouplie ici (V1 : priorite a une version qui
    # marche), borne haute conservee (garde-fou contre l'explosion
    # combinatoire, chapitre 2).
    if not (1 <= len(Z) <= 500):
        print(f"\n!!! ATTENTION : |Z|={len(Z)} hors de la fourchette geree [1,500] !!!")
    else:
        write_domain_yaml(concepts, prereqs)
        print(f"\nEcrit : {OUT_PATH}")
