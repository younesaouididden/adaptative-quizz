"""
Demo Streamlit du quiz adaptatif : reutilise directement kst_engine.py et
domains/piste_b.yaml (7 concepts, 35 questions ecrites a la main) -- aucune
reimplementation, le moteur qui tourne ici est exactement celui teste dans
test_kst_engine.py.

Domaine piste B (cf. ADDENDUM_BANQUE_QUESTIONS.md, domains/piste_b.yaml,
note_calibration.md) : guess = 1/nb_options est une borne combinatoire
conservatrice (guess <= 1/k pour un QCM a k options), pas une valeur choisie ;
slip = 0.10 n'a pas de justification de premier principe, c'est un point sur
un axe a balayer, pas une "valeur experte" figee. Ni l'un ni l'autre n'est le
resultat d'une calibration EM sur des donnees reelles -- celle-ci (piste A,
data/domain.yaml) est un pipeline separe, sur un domaine a 5 concepts issu
des logs Junyi Academy.
"""

from pathlib import Path

import numpy as np
import streamlit as st

from domains.loader import load_domain_yaml
from kst_engine import (
    bayes_update,
    concept_marginals,
    entropy,
    pi_star,
    should_stop,
    uniform_prior,
)

DOMAIN_PATH = Path(__file__).resolve().parent / "domains" / "piste_b.yaml"


@st.cache_resource
def load_domain():
    return load_domain_yaml(DOMAIN_PATH)


def init_state(domain):
    st.session_state.stage = "intro"
    st.session_state.p = uniform_prior(domain)
    st.session_state.asked = set()
    st.session_state.current_q = None
    st.session_state.feedback = None
    st.session_state.h0 = entropy(uniform_prior(domain))


def pick_next(domain):
    q, ig = pi_star(st.session_state.p, domain, st.session_state.asked)
    if q is None or should_stop(st.session_state.p, domain, st.session_state.asked, ig=ig):
        st.session_state.stage = "results"
        st.session_state.current_q = None
    else:
        st.session_state.stage = "quiz"
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
    "Moteur KST &middot; domaine piste B (démo, paramètres experts)</p>",
    unsafe_allow_html=True,
)
st.title("Diagnostic adaptatif")

if st.session_state.stage == "intro":
    st.write(
        "Chaque question réduit l'incertitude sur ce que tu maîtrises. "
        "Pas de score final — un diagnostic, concept par concept."
    )
    st.write(
        "7 domaines de maths (arithmétique de base, fractions/ratios, algèbre "
        "linéaire, algèbre avancée, géométrie, géométrie analytique, "
        "probabilités/statistiques). Le moteur choisit à chaque tour la "
        "question qui t'apprend le plus sur toi."
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
        "Sélection par gain d'information sur Δ(Z), mise à jour bayésienne (BLIM). "
        "Domaine piste B : guess = 1/nb d'options est une borne combinatoire "
        "conservatrice, slip = 0,10 un point sur un axe balayé — ni l'un ni "
        "l'autre n'est une calibration empirique."
    )
    if st.button("Recommencer"):
        init_state(domain)
        st.rerun()
