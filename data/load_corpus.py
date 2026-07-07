"""
Chargement du corpus depuis un fichier CSV.

Format attendu (en-tête obligatoire) :
    id,text,source,categorie

- id        : identifiant unique du chunk (utilisé tel quel comme id ChromaDB)
- text      : le texte du chunk, celui qui sera encodé et retourné au LLM
- source    : d'où vient l'information (nom du carnet, du document, etc.)
- categorie : étiquette libre, utilisable ensuite pour filtrer la recherche
"""

import csv
from pathlib import Path


def load_corpus(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus introuvable : {csv_path}")

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"id", "text", "source", "categorie"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {csv_path} : {sorted(missing)}"
            )

        return list(reader)
