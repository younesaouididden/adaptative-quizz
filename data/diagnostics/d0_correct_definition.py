"""
Diagnostic D0 (DIAGNOSTIC_GUESS_A2.md) : comment 'correct' est-il calcule
aujourd'hui, et a quel point les logs bruts contiennent-ils des tentatives
repetees / des indices ?

Lecture memoire-legere : PAS de group-by sur (user_id, exercise) (risque
sur une machine a 8 Go de RAM dont ~0,6 Go libre). A la place, on accumule
un histogramme de la colonne 'problem_number' (peu de valeurs distinctes,
donc peu couteux) : le nombre de lignes a problem_number==1 donne le
nombre de paires (etudiant, exercice) distinctes sans aucun group-by,
puisque chaque paire ne peut avoir qu'UNE seule ligne a problem_number==1
(en supposant une numerotation sequentielle a partir de 1, comme decrit
dans le README Junyi -- verifie separement par un sondage, cf.
spot_check_sequential()).

Ne modifie AUCUN fichier de production (domain.yaml, responses.parquet).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
LOG_PATH = RAW_DIR / "junyi_ProblemLog_original.csv"

USE_COLS = ["user_id", "exercise", "problem_number", "hint_used", "count_hints"]
CHUNK_SIZE = 500_000


def accumulate_problem_number_histogram(log_path: Path = LOG_PATH,
                                        chunk_size: int = CHUNK_SIZE) -> dict:
    """Accumule, sans jamais grouper par (user_id, exercise) :
    - l'histogramme de problem_number (peu de valeurs distinctes)
    - le nombre total de lignes
    - le nombre de lignes avec hint_used=true
    """
    pn_counts = pd.Series(dtype="int64")
    n_rows = 0
    n_hint = 0

    reader = pd.read_csv(log_path, usecols=USE_COLS,
                         dtype={"exercise": str, "hint_used": str},
                         chunksize=chunk_size)
    for chunk in reader:
        n_rows += len(chunk)
        n_hint += int(chunk["hint_used"].str.lower().eq("true").sum())
        counts = chunk["problem_number"].value_counts()
        pn_counts = pn_counts.add(counts, fill_value=0)

    return {"pn_counts": pn_counts.sort_index(), "n_rows": n_rows, "n_hint": n_hint}


def spot_check_sequential(log_path: Path = LOG_PATH, n_candidates: int = 30,
                          candidate_scan_rows: int = CHUNK_SIZE,
                          chunk_size: int = CHUNK_SIZE) -> dict:
    """Verifie que problem_number forme une sequence 1..N sans trou, sur
    l'historique COMPLET (pas un fragment) d'un petit nombre de paires
    (etudiant, exercice).

    Le fichier est ordonne dans le temps (pas regroupe par etudiant) : les
    lignes d'une meme paire sont dispersees sur tout le fichier. Un premier
    sondage qui se contentait de lire un seul chunk contigu ne voyait donc
    qu'un FRAGMENT tronque de chaque historique -- methode invalidee,
    corrigee ici en deux passes :
      1) choisir n_candidates paires ayant repete dans les
         candidate_scan_rows premieres lignes (peu couteux -- notez que
         cette fenetre peut elle-meme rater des paires dont les deux
         premieres occurrences sont deja tres eloignees ; ce n'est qu'un
         echantillonnage, pas une garantie d'exhaustivite) ;
      2) rebalayer TOUT le fichier par chunks de chunk_size lignes, en ne
         gardant que les lignes de CES paires precises, pour reconstruire
         leur historique complet quelle que soit sa dispersion.
    """
    first_chunk = pd.read_csv(log_path, usecols=["user_id", "exercise", "problem_number"],
                              nrows=candidate_scan_rows)
    counts = first_chunk.groupby(["user_id", "exercise"]).size()
    candidates = set(counts[counts > 1].index[:n_candidates])
    if not candidates:
        return {"n_paires_testees": 0, "n_sequentielles_1_a_N": 0}

    cand_users = {u for u, _ in candidates}
    collected: dict[tuple, list[int]] = {c: [] for c in candidates}

    reader = pd.read_csv(log_path, usecols=["user_id", "exercise", "problem_number"],
                         chunksize=chunk_size)
    for chunk in reader:
        sub = chunk[chunk["user_id"].isin(cand_users)]
        if sub.empty:
            continue
        for row in sub.itertuples(index=False):
            key = (row.user_id, row.exercise)
            if key in collected:
                collected[key].append(row.problem_number)

    n_sequential = sum(1 for seq in collected.values()
                      if sorted(seq) == list(range(1, len(seq) + 1)))
    return {"n_paires_testees": len(collected), "n_sequentielles_1_a_N": n_sequential,
           "exemples_non_sequentiels": [
               (k, sorted(v)) for k, v in list(collected.items())
               if sorted(v) != list(range(1, len(v) + 1))][:3]}


if __name__ == "__main__":
    print("--- Sondage de sequentialite de problem_number (historique complet, 2 passes) ---")
    sc = spot_check_sequential()
    print(sc)
    print()

    print("--- Histogramme complet (fichier entier, streaming) ---")
    hist = accumulate_problem_number_histogram()
    pn_counts = hist["pn_counts"]
    n_rows, n_pairs = hist["n_rows"], int(pn_counts.get(1, 0))

    print(f"lignes totales                    : {n_rows:,}")
    print(f"paires (etudiant,exercice) distinctes (= lignes a problem_number=1) : {n_pairs:,}")
    print(f"tentatives moyennes par paire      : {n_rows / n_pairs:.2f}")
    print(f"part de lignes avec indice utilise : {100 * hist['n_hint'] / n_rows:.2f}%")
    print(f"problem_number maximum observe     : {int(pn_counts.index.max())}")
    print()
    print("part des paires atteignant au moins N tentatives :")
    # nb de paires avec >= N tentatives = nb de lignes avec problem_number == N
    # (puisque chaque paire produit exactement une ligne par valeur de N
    # jusqu'a son propre maximum)
    for n in (1, 2, 3, 5, 10, 20, 50, 100):
        # nb de paires ayant AU MOINS N tentatives = nb de lignes a
        # problem_number==N (une paire avec exactement N tentatives
        # contribue une ligne a chaque valeur 1..N, dont une a la valeur N)
        pct = 100 * int(pn_counts.get(n, 0)) / n_pairs if n_pairs else float("nan")
        print(f"  >= {n:3d} tentatives : {pct:6.2f}% des paires")
