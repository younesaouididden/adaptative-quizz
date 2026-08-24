"""
Moteur de quiz adaptatif fonde sur la Knowledge Space Theory.

Implemente les couches 1, 2 et 4 du PFA :
  1. Combinatoire : construction de Z a partir d'un graphe de prerequis
  2. Probabilite  : croyance p sur Delta(Z), mise a jour bayesienne (BLIM)
  4. Controle     : selection par maximisation du gain d'information

Aucune dependance hors numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, combinations

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# COUCHE 1 : L'espace des connaissances Z (chapitre 2)
# ---------------------------------------------------------------------------

def build_knowledge_space(items: list[str],
                          prereqs: list[tuple[str, str]]) -> list[frozenset]:
    """Construit Z = { z subset P : z clos vers le bas pour la relation de surmise }.

    prereqs est une liste de paires (a, b) signifiant "a est prerequis de b",
    c'est-a-dire a <= b. On calcule d'abord la cloture transitive, puis on
    retient les sous-ensembles clos.

    Le resultat est automatiquement un espace de connaissance (union-closed) :
    l'intersection et l'union de deux ensembles clos sont closes.

    P = les CONCEPTS du domaine (pas les questions) : Z se construit une
    fois pour toutes sur le graphe de prerequis, independamment de la
    taille de la banque de questions.
    """
    # cloture transitive : required[b] = tous les prerequis de b
    required = {i: set() for i in items}
    for a, b in prereqs:
        required[b].add(a)
    changed = True
    while changed:                      # Floyd-Warshall du pauvre
        changed = False
        for b in items:
            for a in list(required[b]):
                new = required[a] - required[b]
                if new:
                    required[b] |= new
                    changed = True

    for i in items:
        if i in required[i]:
            raise ValueError(f"Cycle de prerequis detecte sur l'item '{i}'")

    powerset = chain.from_iterable(combinations(items, r)
                                   for r in range(len(items) + 1))
    Z = [frozenset(s) for s in powerset
         if all(required[x] <= set(s) for x in s)]
    return sorted(Z, key=lambda z: (len(z), sorted(z)))


# ---------------------------------------------------------------------------
# COUCHE 2 : Le modele de reponse (BLIM) et la croyance (chapitre 3)
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    """Noeud du graphe de prerequis, element de P. Z se construit sur les
    concepts, jamais sur les questions."""
    name: str


@dataclass
class Question:
    """Item de la banque de questions, rattache a un concept (chapitre 3).

    slip = erreur d'inattention, guess = reponse au hasard. Plusieurs
    questions peuvent partager le meme concept avec des (slip, guess)
    differents : c'est ce qui donne un vrai choix a la selection gloutonne
    (chapitres 6-7) une fois qu'un concept a deja ete interroge une fois.

    Contrainte de validite du BLIM : slip + guess < 1, sinon repondre juste
    est une preuve de NON-maitrise et l'inference s'inverse.
    """
    id: str
    concept: str
    slip: float = 0.10
    guess: float = 0.20
    enonce: str = ""

    def __post_init__(self):
        if self.slip + self.guess >= 1.0:
            raise ValueError(f"Question '{self.id}': slip + guess doit etre < 1")


@dataclass
class Domain:
    concepts: list[Concept]
    prereqs: list[tuple[str, str]]
    questions: list[Question]
    Z: list[frozenset] = field(init=False)

    def __post_init__(self):
        names = [c.name for c in self.concepts]
        concept_names = set(names)
        for q in self.questions:
            if q.concept not in concept_names:
                raise ValueError(
                    f"Question '{q.id}' rattachee a un concept inconnu '{q.concept}'")
        self.Z = build_knowledge_space(names, self.prereqs)
        # matrice de vraisemblance L[z, q] = P(correct | z, question), chapitre 3.
        # Indexee sur les QUESTIONS, pas les concepts : deux questions du meme
        # concept partagent la meme condition "concept in z" mais ont chacune
        # leur propre (slip, guess) -> colonnes distinctes.
        self.L = np.array([[self._p_correct(z, q) for q in self.questions]
                           for z in self.Z])

    @staticmethod
    def _p_correct(z: frozenset, question: Question) -> float:
        return (1.0 - question.slip) if question.concept in z else question.guess

    @property
    def n_states(self) -> int:
        return len(self.Z)

    @property
    def n_questions(self) -> int:
        return len(self.questions)


def uniform_prior(domain: Domain) -> np.ndarray:
    return np.full(domain.n_states, 1.0 / domain.n_states)


def bayes_update(p: np.ndarray, domain: Domain, q: int, correct: bool) -> np.ndarray:
    """Mise a jour bayesienne p_{t+1}(z) ~ P(y|a,z) p_t(z), en log pour la stabilite.

    q indexe domain.questions (pas domain.concepts).

    Chapitres 4-5 : le bord du simplexe est a distance infinie, un p(z) qui
    atteint exactement 0 y est pieoge de facon irreversible. On lisse donc p
    ET la vraisemblance avant de combiner (jamais un seul des deux), puis on
    renormalise par la somme reelle -- ce qui est algebriquement identique a
    la formule p <- (p+eps)/(1+|Z|*eps) mais plus robuste aux erreurs
    d'arrondi flottant.
    """
    lik = domain.L[:, q] if correct else 1.0 - domain.L[:, q]
    log_post = np.log(p + EPS) + np.log(lik + EPS)
    log_post -= log_post.max()                # stabilisation
    post = np.exp(log_post)
    return post / post.sum()


def concept_marginals(p: np.ndarray, domain: Domain) -> dict[str, float]:
    """P(concept c maitrise) = somme des p(z) sur les z contenant c (chapitre 3).

    C'est CA qu'on affiche a l'utilisateur, jamais p en entier (chapitre 6).
    """
    return {c.name: float(sum(p[k] for k, z in enumerate(domain.Z)
                              if c.name in z))
            for c in domain.concepts}


# ---------------------------------------------------------------------------
# COUCHE 4 : Gain d'information et selection (chapitres 6-7)
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    """H(p) en bits.

    On filtre p>EPS plutot que de lisser : lim x->0 (x log x) = 0, donc
    ignorer les etats quasi nuls est exact, pas une approximation biaisee.
    """
    q = p[p > EPS]
    return float(-np.sum(q * np.log2(q)))


def information_gain_exact(p: np.ndarray, domain: Domain, q: int) -> float:
    """IG(a;p) = H(p) - E_y[ H(posterior) ].

    Identique a E_y[ KL(posterior || p) ] et a l'information mutuelle I(Z;Y|a),
    mais la version entropie evite tout calcul explicite de KL -- donc evite
    le piege de smoothing asymetrique du point suivant. C'est pour ca qu'on
    l'implemente ainsi plutot que la formule KL directement (chapitre 6).

    q indexe domain.questions. Cout O(|Z|) : avec |Y|=2 la somme exacte
    suffit, pas besoin de Monte Carlo.
    """
    lik = domain.L[:, q]
    p_correct = float(np.dot(p, lik))       # predictive P(y=1 | a, p)
    ig = entropy(p)
    for prob_y, l in ((p_correct, lik), (1.0 - p_correct, 1.0 - lik)):
        if prob_y <= EPS:
            continue
        post = p * l
        post = post / post.sum()
        ig -= prob_y * entropy(post)
    return ig


def information_gain_mc(p: np.ndarray, domain: Domain, q: int,
                        n_samples: int = 200,
                        rng: np.random.Generator | None = None) -> float:
    """Estimateur Monte Carlo de IG via E_y[ KL(posterior || p) ] (chapitres 4-7).

    Ici il est inutile (|Y| = 2), mais on le garde pour verifier empiriquement
    la convergence vers la valeur exacte -- et il devient indispensable des
    que les reponses sont a choix multiples ou que |Z| est grand.

    Piege (chapitre 4-5) : lisser SEULEMENT le denominateur du KL,
    log(P / (Q+eps)), biaise l'estimateur et peut le rendre negatif. On
    lisse donc les deux distributions avec la meme formule
    p <- (p+eps)/(1+|Z|*eps) avant de calculer le KL.
    """
    rng = rng or np.random.default_rng()
    p_correct = float(np.dot(p, domain.L[:, q]))
    ys = rng.random(n_samples) < p_correct
    n_z = len(p)

    def smooth(dist: np.ndarray) -> np.ndarray:
        return (dist + EPS) / (1.0 + n_z * EPS)

    p_s = smooth(p)
    total = 0.0
    for y in ys:
        post = bayes_update(p, domain, q, bool(y))
        post_s = smooth(post)
        total += float(np.sum(post_s * np.log2(post_s / p_s)))
    return total / n_samples


def select_next(p: np.ndarray, domain: Domain,
                asked: set[int]) -> tuple[int, float]:
    """Politique gloutonne : argmax du gain d'information sur les questions non posees.

    asked contient des indices de QUESTIONS (pas de concepts) : plusieurs
    questions du meme concept restent eligibles tant qu'elles n'ont pas
    elles-memes ete posees. C'est le coeur du refactor concept/question --
    sans banque de plusieurs questions par concept, argmax n'a rien a
    departager une fois chaque concept interroge une fois.
    """
    best, best_ig = None, -np.inf
    for q in range(domain.n_questions):
        if q in asked:
            continue
        ig = information_gain_exact(p, domain, q)
        if ig > best_ig:
            best, best_ig = q, ig
    return best, best_ig


def should_stop(p: np.ndarray, domain: Domain, asked: set[int],
                max_questions: int = 30,
                confidence: float = 0.85,
                min_ig: float = 0.01) -> bool:
    """Trois criteres d'arret (chapitre 7), le premier qui se declenche gagne."""
    if len(asked) >= min(max_questions, domain.n_questions):
        return True
    if p.max() >= confidence:                       # un etat domine
        return True
    _, ig = select_next(p, domain, asked)           # plus rien a apprendre
    return ig < min_ig


# ---------------------------------------------------------------------------
# SIMULATION : etudiant synthetique, adaptatif vs aleatoire
# ---------------------------------------------------------------------------

def simulate(domain: Domain, z_true: frozenset, adaptive: bool = True,
             seed: int = 0, max_questions: int = 30):
    """Fait passer le quiz a un etudiant simule dont on connait l'etat reel."""
    rng = np.random.default_rng(seed)
    p = uniform_prior(domain)
    asked: set[int] = set()
    trace = [entropy(p)]

    while not should_stop(p, domain, asked, max_questions=max_questions):
        if adaptive:
            q, _ = select_next(p, domain, asked)
        else:
            q = int(rng.choice([i for i in range(domain.n_questions)
                                if i not in asked]))
        question = domain.questions[q]
        # l'etudiant repond selon le vrai BLIM
        true_p = (1 - question.slip) if question.concept in z_true else question.guess
        correct = bool(rng.random() < true_p)
        p = bayes_update(p, domain, q, correct)
        asked.add(q)
        trace.append(entropy(p))

    z_hat = domain.Z[int(np.argmax(p))]
    return {"n_questions": len(asked), "entropy_trace": trace,
            "z_hat": z_hat, "correct_diagnosis": z_hat == z_true,
            "confidence": float(p.max()), "belief": p}


# ---------------------------------------------------------------------------
# DOMAINE JOUET -- a remplacer par le tien
# ---------------------------------------------------------------------------

_CONCEPT_NAMES = ["fractions", "equations1", "fonctions", "derivees",
                  "limites", "integrales", "proba_base", "var_aleatoire"]

_PREREQS = [
    ("fractions", "equations1"),
    ("equations1", "fonctions"),
    ("fonctions", "limites"),
    ("limites", "derivees"),
    ("derivees", "integrales"),
    ("fractions", "proba_base"),
    ("proba_base", "var_aleatoire"),
    ("fonctions", "var_aleatoire"),
]

_BASE_SLIP_GUESS = {
    "fractions":     (0.05, 0.25),
    "equations1":    (0.10, 0.20),
    "fonctions":     (0.10, 0.20),
    "derivees":      (0.12, 0.20),
    "limites":       (0.12, 0.25),
    "integrales":    (0.15, 0.15),
    "proba_base":    (0.08, 0.25),
    "var_aleatoire": (0.15, 0.20),
}


def _question_bank(concept: str, base_slip: float, base_guess: float,
                   n: int) -> list[Question]:
    """n questions synthetiques pour un concept, avec un slip/guess legerement
    varie autour de la valeur de base -- purement pour le domaine jouet.

    Le vrai domaine chargera sa banque depuis un fichier YAML/JSON, pas
    depuis du code en dur (voir PROMPT_DEMARRAGE section 8).
    """
    offsets = [0.0, 0.02, -0.02, 0.04, -0.04, 0.03, -0.03, 0.01, -0.01, 0.015][:n]
    return [Question(id=f"{concept}_q{i+1}", concept=concept,
                     slip=round(min(0.30, max(0.01, base_slip + off)), 3),
                     guess=round(min(0.35, max(0.05, base_guess - off)), 3))
           for i, off in enumerate(offsets)]


def make_demo_domain(questions_per_concept: int = 3) -> Domain:
    """Domaine jouet parametre par la taille de la banque de questions --
    sert a comparer le comportement avant/apres le refactor
    concept/question (1 question/concept vs plusieurs)."""
    return Domain(
        concepts=[Concept(n) for n in _CONCEPT_NAMES],
        prereqs=_PREREQS,
        questions=[q for name in _CONCEPT_NAMES
                  for q in _question_bank(name, *_BASE_SLIP_GUESS[name],
                                          n=questions_per_concept)],
    )


DEMO = make_demo_domain(questions_per_concept=3)


if __name__ == "__main__":
    d = DEMO
    n_concepts = len(d.concepts)
    print(f"Concepts : {n_concepts}   |   2^n = {2**n_concepts}"
          f"   |   |Z| = {d.n_states}   "
          f"({100*d.n_states/2**n_concepts:.0f}% du power set)")
    print(f"Questions : {d.n_questions}   (banque de "
          f"{d.n_questions // n_concepts} par concept)")
    print(f"Entropie initiale H(p0) = {entropy(uniform_prior(d)):.2f} bits\n")

    # verification de l'exemple numerique de la presentation 3
    print("--- Test : reproduction de l'exemple du collegue ---")
    toy = Domain(concepts=[Concept("integrales")], prereqs=[],
                questions=[Question("integrales_q1", "integrales",
                                    slip=0.10, guess=0.20)])
    p_toy = uniform_prior(toy)
    p_toy = bayes_update(p_toy, toy, 0, correct=True)
    print(f"prior 0.500 -> posterior {concept_marginals(p_toy, toy)['integrales']:.3f}"
          f"   (attendu 0.818)\n")

    # coherence Monte Carlo vs exact (avec le smoothing KL corrige)
    print("--- Test : Monte Carlo converge vers l'exact ---")
    p0 = uniform_prior(d)
    for q in (0, 4, d.n_questions - 1):
        ex = information_gain_exact(p0, d, q)
        mc = information_gain_mc(p0, d, q, n_samples=5000,
                                 rng=np.random.default_rng(1))
        print(f"  {d.questions[q].id:16s}  exact={ex:.4f}  MC={mc:.4f}")

    # benchmark AVANT/APRES le refactor : 1 question/concept vs plusieurs
    print("\n--- Benchmark : avant (1Q/concept) vs apres (3Q/concept) le refactor ---")
    for n_q_per_concept in (1, 3):
        dom = make_demo_domain(questions_per_concept=n_q_per_concept)
        print(f"\n  banque = {n_q_per_concept} question(s)/concept "
              f"(|A| = {dom.n_questions}, |Z| = {dom.n_states})")
        for label, adaptive in (("adaptatif", True), ("aleatoire", False)):
            n_q, acc = [], []
            for seed, z in enumerate(dom.Z):
                r = simulate(dom, z, adaptive=adaptive, seed=seed)
                n_q.append(r["n_questions"])
                acc.append(r["correct_diagnosis"])
            print(f"    {label:10s} : {np.mean(n_q):.2f} questions en moyenne, "
                  f"diagnostic exact {100*np.mean(acc):.0f}%")
