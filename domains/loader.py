"""Chargeur generique pour les domaines piste B (config YAML, schema B1).

Piste B : concepts + questions ecrits a la main, slip/guess experts non
calibres (cf. ADDENDUM_BANQUE_QUESTIONS.md). Separe de kst_engine.py pour
garder le moteur sans dependance hors numpy.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kst_engine import Concept, Domain, Question


def load_domain_yaml(path: str | Path) -> tuple[Domain, list[dict], dict[str, str]]:
    """Charge un domaine + banque de questions depuis un YAML au format B1.

    Retourne (domain, meta, labels) :
      - domain : objet moteur, pret pour bayes_update/select_next
      - meta   : liste alignee sur domain.questions (stem/options/answer/difficulty)
      - labels : id de concept -> libelle humain
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    concepts = [Concept(c["id"]) for c in data["concepts"]]
    labels = {c["id"]: c.get("label", c["id"]) for c in data["concepts"]}
    slip_guess = {c["id"]: (c["slip"], c["guess"]) for c in data["concepts"]}
    prereqs = [tuple(p) for p in data["prerequisites"]]

    questions, meta = [], []
    for q in data["questions"]:
        slip, guess = slip_guess[q["concept"]]
        questions.append(Question(q["id"], q["concept"], slip=slip, guess=guess))
        meta.append({
            "stem": q["stem"],
            "options": q["options"],
            "answer": q["answer"],
            "difficulty": q.get("difficulty"),
        })

    domain = Domain(concepts=concepts, prereqs=prereqs, questions=questions)
    return domain, meta, labels
