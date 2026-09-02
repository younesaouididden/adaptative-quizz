"""Tests du chargeur/validateur piste B, sur fixtures YAML synthetiques."""

import textwrap

import pytest
import yaml

from domains.loader import load_domain_yaml
from domains.validate import validate

MINI_DOMAIN = textwrap.dedent("""\
    schema_version: 1
    domain: test_mini
    concepts:
      - id: a
        label: Concept A
        slip: 0.10
        guess: 0.25
        calibrated: false
      - id: b
        label: Concept B
        slip: 0.10
        guess: 0.25
        calibrated: false
    prerequisites:
      - [a, b]
    questions:
      - id: a_1
        concept: a
        difficulty: 1
        stem: "Question A1 ?"
        options: ["x", "y", "z", "w"]
        answer: 0
      - id: a_2
        concept: a
        difficulty: 2
        stem: "Question A2 ?"
        options: ["x", "y", "z", "w"]
        answer: 1
      - id: a_3
        concept: a
        difficulty: 3
        stem: "Question A3 ?"
        options: ["x", "y", "z", "w"]
        answer: 2
      - id: b_1
        concept: b
        difficulty: 1
        stem: "Question B1 ?"
        options: ["x", "y", "z", "w"]
        answer: 3
      - id: b_2
        concept: b
        difficulty: 2
        stem: "Question B2 ?"
        options: ["x", "y", "z", "w"]
        answer: 0
      - id: b_3
        concept: b
        difficulty: 3
        stem: "Question B3 ?"
        options: ["x", "y", "z", "w"]
        answer: 1
    """)


@pytest.fixture
def mini_yaml(tmp_path):
    p = tmp_path / "mini.yaml"
    p.write_text(MINI_DOMAIN, encoding="utf-8")
    return p


def test_load_domain_yaml_builds_engine_domain(mini_yaml):
    domain, meta, labels = load_domain_yaml(mini_yaml)

    assert [c.name for c in domain.concepts] == ["a", "b"]
    assert len(domain.questions) == 6
    assert labels == {"a": "Concept A", "b": "Concept B"}
    # a est prerequis de b -> {b} seul seul n'est pas un etat valide
    assert frozenset({"b"}) not in domain.Z
    assert frozenset({"a", "b"}) in domain.Z


def test_load_domain_yaml_meta_aligned_with_questions(mini_yaml):
    domain, meta, labels = load_domain_yaml(mini_yaml)

    for q, m in zip(domain.questions, meta):
        assert m["stem"]
        assert len(m["options"]) == 4
        assert 0 <= m["answer"] < 4


def test_load_domain_yaml_questions_inherit_concept_slip_guess(mini_yaml):
    domain, meta, labels = load_domain_yaml(mini_yaml)

    for q in domain.questions:
        assert q.slip == 0.10
        assert q.guess == 0.25


def test_validate_accepts_well_formed_domain(mini_yaml):
    assert validate(str(mini_yaml)) == []


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


def test_validate_rejects_too_few_questions_per_concept(tmp_path):
    data = yaml.safe_load(MINI_DOMAIN)
    data["questions"] = [q for q in data["questions"] if q["id"] != "b_3"]
    p = _write(tmp_path, "broken.yaml", data)

    errors = validate(str(p))
    assert any("minimum 3" in e for e in errors)


def test_validate_rejects_duplicate_question_ids(tmp_path):
    data = yaml.safe_load(MINI_DOMAIN)
    data["questions"][3]["id"] = "a_1"  # b_1 -> id deja pris par a_1
    p = _write(tmp_path, "broken.yaml", data)

    errors = validate(str(p))
    assert any("dupliques" in e for e in errors)


def test_validate_rejects_invalid_answer_index(tmp_path):
    data = yaml.safe_load(MINI_DOMAIN)
    data["questions"][0]["answer"] = 9
    p = _write(tmp_path, "broken.yaml", data)

    errors = validate(str(p))
    assert any("index de reponse invalide" in e for e in errors)


def test_validate_rejects_prereq_cycle(tmp_path):
    data = yaml.safe_load(MINI_DOMAIN)
    data["prerequisites"].append(["b", "a"])
    p = _write(tmp_path, "cyclic.yaml", data)

    errors = validate(str(p))
    assert any("Cycle" in e for e in errors)
