"""
Demo Streamlit du quiz adaptatif : reutilise directement kst_engine.py et
data/domain.yaml (5 concepts, slip/guess calibres par EM sur les logs reels
Junyi Academy, tache A2) -- aucune reimplementation, le moteur qui tourne
ici est exactement celui teste dans test_kst_engine.py.

La banque de questions (QUESTION_BANK ci-dessous) est ecrite a la main :
Junyi ne fournit que des logs d'interaction, pas d'enonces (piste B,
authorship de questions, en pause -- cf. ADDENDUM_BANQUE_QUESTIONS.md).
Chaque question herite du slip/guess calibre de son concept.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import yaml

from kst_engine import (
    Concept,
    Domain,
    Question,
    bayes_update,
    concept_marginals,
    entropy,
    select_next,
    should_stop,
    uniform_prior,
)

DOMAIN_PATH = Path(__file__).resolve().parent / "data" / "domain.yaml"

QUESTION_BANK = {
    "arithmetic": [
        ("Combien font 7 × 8 ?", ["54", "56", "64", "72"], 1),
        ("Que vaut 3/4 + 1/8 ?", ["7/8", "1/2", "5/8", "1"], 0),
        ("15 % de 200, ça fait combien ?", ["15", "20", "30", "40"], 2),
        ("Simplifie 24/36.", ["3/4", "2/3", "4/6", "1/2"], 1),
    ],
    "algebra": [
        ("Résous : 2x + 5 = 13", ["x = 3", "x = 4", "x = 5", "x = 9"], 1),
        ("Factorise x² − 9.", ["(x−3)(x+3)", "(x−9)(x+1)", "(x−3)²", "(x+3)²"], 0),
        ("Résous : 3x − 7 = 2x + 1", ["x = 6", "x = 7", "x = 8", "x = −8"], 2),
        ("Développe (x+2)(x−2).", ["x² − 4", "x² + 4", "x² − 4x + 4", "x² + 4x − 4"], 0),
    ],
    "geometry": [
        ("Aire d'un rectangle 5 × 3 ?", ["8", "15", "16", "30"], 1),
        ("Somme des angles d'un triangle ?", ["90°", "180°", "270°", "360°"], 1),
        ("Périmètre d'un carré de côté 6 ?", ["12", "18", "24", "36"], 2),
        ("Aire d'un cercle de rayon 2 (en fonction de π) ?", ["2π", "4π", "8π", "16π"], 1),
    ],
    "analytic_geometry": [
        ("Pente de la droite passant par (0,0) et (2,4) ?", ["1", "2", "4", "1/2"], 1),
        ("Distance entre (0,0) et (3,4) ?", ["4", "5", "6", "7"], 1),
        ("Équation de la droite de pente 2 passant par l'origine ?",
         ["y = 2x", "y = x + 2", "y = 2x + 1", "x = 2y"], 0),
    ],
    "probability_statistics": [
        ("Probabilité d'obtenir Pile avec une pièce équilibrée ?", ["1/4", "1/3", "1/2", "1"], 2),
        ("Moyenne de {2, 4, 6} ?", ["3", "4", "5", "6"], 1),
        ("Probabilité d'obtenir un 6 avec un dé équilibré ?", ["1/2", "1/4", "1/6", "1/8"], 2),
    ],
}


@st.cache_resource
def load_domain():
    with open(DOMAIN_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    concepts = [Concept(c["id"]) for c in data["concepts"]]
    labels = {c["id"]: c.get("label", c["id"]) for c in data["concepts"]}
    slip_guess = {c["id"]: (c["slip"], c["guess"]) for c in data["concepts"]}
    prereqs = [tuple(p) for p in data["prerequisites"]]

    questions, meta = [], []
    for concept_id, items in QUESTION_BANK.items():
        slip, guess = slip_guess[concept_id]
        for i, (stem, options, answer) in enumerate(items):
            questions.append(Question(f"{concept_id}_{i + 1}", concept_id, slip=slip, guess=guess))
            meta.append({"stem": stem, "options": options, "answer": answer})

    domain = Domain(concepts=concepts, prereqs=prereqs, questions=questions)
    return domain, meta, labels


def init_state(domain):
    st.session_state.stage = "intro"
    st.session_state.p = uniform_prior(domain)
    st.session_state.asked = set()
    st.session_state.current_q = None
    st.session_state.feedback = None
    st.session_state.h0 = entropy(uniform_prior(domain))


def pick_next(domain):
    q, ig = select_next(st.session_state.p, domain, st.session_state.asked)
    if q is None or should_stop(st.session_state.p, domain, st.session_state.asked, ig=ig):
        st.session_state.stage = "results"
        st.session_state.current_q = None
    else:
        st.session_state.current_q = q


def answer(domain, meta, q_idx, chosen_idx):
    correct = chosen_idx == meta[q_idx]["answer"]
    st.session_state.feedback = (correct, meta[q_idx]["options"][meta[q_idx]["answer"]])
    st.session_state.p = bayes_update(st.session_state.p, domain, q_idx, correct)
    st.session_state.asked.add(q_idx)
    pick_next(domain)


st.set_page_config(page_title="Diagnostic Adaptatif", page_icon="\U0001f9ed", layout="centered")

domain, meta, labels = load_domain()
if "stage" not in st.session_state:
    init_state(domain)

st.markdown(
    "<p style='font-family:monospace;color:#0E7C7B;font-size:0.78rem;"
    "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:-0.5rem;'>"
    "Moteur KST &middot; domaine calibré sur Junyi Academy</p>",
    unsafe_allow_html=True,
)
st.title("Diagnostic adaptatif")

if st.session_state.stage == "intro":
    st.write(
        "Chaque question réduit l'incertitude sur ce que tu maîtrises. "
        "Pas de score final — un diagnostic, concept par concept."
    )
    st.write(
        "5 domaines de maths (arithmétique, algèbre, géométrie, "
        "géométrie analytique, probabilités/statistiques). Le moteur choisit "
        "à chaque tour la question qui t'apprend le plus sur toi."
    )
    if st.button("Commencer le diagnostic", type="primary"):
        pick_next(domain)
        st.rerun()

elif st.session_state.stage == "quiz":
    q_idx = st.session_state.current_q
    q_meta = meta[q_idx]
    question = domain.questions[q_idx]

    h = entropy(st.session_state.p)
    pct = max(0.0, min(1.0, 1 - h / st.session_state.h0))
    st.caption(f"Question {len(st.session_state.asked) + 1} — {pct * 100:.0f} % d'incertitude levée")
    st.progress(pct)

    st.markdown(f"**{labels[question.concept]}**")
    st.subheader(q_meta["stem"])

    if st.session_state.feedback is not None:
        correct, right_answer = st.session_state.feedback
        if correct:
            st.success("Correct.")
        else:
            st.error(f"Incorrect — la bonne réponse était « {right_answer} ».")
        st.session_state.feedback = None

    cols = st.columns(2)
    for i, opt in enumerate(q_meta["options"]):
        with cols[i % 2]:
            if st.button(opt, key=f"opt_{q_idx}_{i}", use_container_width=True):
                answer(domain, meta, q_idx, i)
                st.rerun()

elif st.session_state.stage == "results":
    p = st.session_state.p
    h = entropy(p)
    conf = float(np.max(p))
    marg = concept_marginals(p, domain)

    c1, c2, c3 = st.columns(3)
    c1.metric("Questions posées", len(st.session_state.asked))
    c2.metric("Bits d'incertitude restants", f"{h:.2f}")
    c3.metric("Confiance", f"{conf * 100:.0f}%")

    st.subheader("Diagnostic par concept")
    for c in domain.concepts:
        m = marg[c.name]
        if m >= 0.6:
            tag = ":green[maîtrisé]"
        elif m <= 0.4:
            tag = ":red[lacune]"
        else:
            tag = ":orange[incertain]"
        st.write(f"**{labels[c.name]}** — {tag} ({m * 100:.0f}%)")
        st.progress(min(1.0, max(0.0, m)))

    best_idx = int(np.argmax(p))
    z_hat = domain.Z[best_idx]
    candidates = [c.name for c in domain.concepts
                 if c.name not in z_hat and (z_hat | {c.name}) in domain.Z]
    if candidates:
        st.info(f"\U0001f9ed À travailler ensuite : **{labels[candidates[0]]}** "
               "(ses prérequis sont couverts d'après ton état estimé le plus probable).")

    st.caption(
        "Sélection par gain d'information sur Δ(Z), mise à jour bayésienne (BLIM), "
        "slip/guess calibrés par EM sur 25,7M réponses réelles (247k étudiants, Junyi Academy)."
    )
    if st.button("Recommencer"):
        init_state(domain)
        st.rerun()
