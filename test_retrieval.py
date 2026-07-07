"""
Test manuel de la brique 1 (étape 3.3 du mini-TP).

Premier run : indexe le corpus depuis data/corpus.csv.
Runs suivants : recharge la base persistée (aucun ré-encodage) et vérifie
que les questions de test remontent le bon chunk en tête.
"""

from config import CORPUS_CSV_PATH
from vector_db import VectorDB

TEST_QUESTIONS = [
    "Quelle est la couleur du chat de Bob ?",
    "Où Henri refuse-t-il de dormir ?",
    "Combien mesure la girafe de Sophie ?",
    "Quel hymne joue le klaxon du vélo de Marc ?",
    "Que sait faire le poisson rouge d'Amélie ?",
    "Que collectionne la grand-mère de Paul ?",
]

if __name__ == "__main__":
    try:
        db = VectorDB()
        print("Base existante rechargée (aucun ré-encodage).")
    except ValueError:
        db = VectorDB(csv_path=CORPUS_CSV_PATH)
        print("Base créée à partir de data/corpus.csv.")

    for q in TEST_QUESTIONS:
        results = db.retrieve(q, n=1)
        top = results[0]
        print(f"\nQ: {q}")
        print(f"  -> {top['text']}  (categorie={top['categorie']}, distance={top['distance']:.4f})")

    print("\n--- Test du filtre par categorie ---")
    results = db.retrieve("Parle-moi d'un objet insolite", n=3, categorie="objets")
    for r in results:
        print(f"  [{r['categorie']}] {r['text']}")
