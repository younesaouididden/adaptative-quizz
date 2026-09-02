"""Validateur de domaine piste B (protocole B2).

Usage : python domains/validate.py domains/piste_b.yaml

Verifie : absence de cycle dans les prerequis, slip+guess<1 par concept, au
moins 3 questions par concept, index de reponse valide, unicite des ids.
Affiche |Z|, 2^n et H(p0) (mesure de la reduction apportee par le graphe de
prerequis par rapport au powerset des concepts).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domains.loader import load_domain_yaml  # noqa: E402
from kst_engine import entropy, uniform_prior  # noqa: E402


def validate(path: str) -> list[str]:
    """Retourne la liste des erreurs trouvees (vide si le domaine est valide)."""
    try:
        domain, meta, labels = load_domain_yaml(path)
    except ValueError as e:
        return [str(e)]

    errors = []

    ids = [q.id for q in domain.questions]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        errors.append(f"ids de question dupliques : {dupes}")

    counts = Counter(q.concept for q in domain.questions)
    for c in domain.concepts:
        if counts[c.name] < 3:
            errors.append(f"concept '{c.name}' : {counts[c.name]} question(s), minimum 3")

    for q, m in zip(domain.questions, meta):
        if not (0 <= m["answer"] < len(m["options"])):
            errors.append(f"question '{q.id}' : index de reponse invalide ({m['answer']})")

    n = len(domain.concepts)
    print(f"concepts        : {n}")
    print(f"|Z|             : {domain.n_states}  (2^{n} = {2 ** n}, "
          f"reduction x{2 ** n / domain.n_states:.1f})")
    print(f"H(p0)           : {entropy(uniform_prior(domain)):.2f} bits")
    print(f"questions/concept : {dict(sorted(counts.items()))}")

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python domains/validate.py <chemin_yaml>")
        sys.exit(1)

    errs = validate(sys.argv[1])
    if errs:
        print("\nErreurs :")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("\nDomaine valide.")
