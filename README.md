# Mon premier RAG

Mini-RAG pédagogique : ChromaDB + sentence-transformers + Groq + agent modérateur.

> Corpus de test volontairement absurde ("Le chat bleu de Bob s'appelle Henri") :
> ces faits n'existent nulle part sur Internet, donc si le système répond juste,
> c'est forcément grâce au retrieval.

## Architecture

```
config.py                          # noms de modèles et chemins, un seul endroit
data/corpus.csv                    # corpus jouet (id, text, source, categorie)
data/load_corpus.py                # lecture et validation du CSV
vector_db.py                       # brique 1 : ChromaDB persisté + sentence-transformers
moderator.py                       # brique 2 : détection de prompt injection (Groq safeguard)
rag.py                             # brique 3 : orchestrateur
prompts/moderator_system_prompt.txt
prompts/rag_system_prompt.txt      # contient {{Chunks}}
test_retrieval.py                  # test manuel de la brique 1 (étape 3.3)
demo.py                            # mise à l'épreuve complète (section 6)
```

## Format du corpus

`data/corpus.csv` doit contenir ces colonnes :

| colonne     | rôle                                                        |
|-------------|--------------------------------------------------------------|
| `id`        | identifiant unique, utilisé tel quel comme id ChromaDB        |
| `text`      | le texte encodé et retourné au LLM                            |
| `source`    | provenance de l'information (nom du document, du carnet...)   |
| `categorie` | étiquette libre, utilisable pour filtrer `retrieve()`         |

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis colle ta clé depuis console.groq.com
```

## Utilisation

```bash
# Indexe data/corpus.csv et vérifie la recherche (crée chroma_db/ au 1er run)
python test_retrieval.py

# Pipeline complet : modération -> retrieval -> génération
python demo.py
```

## Ce que le pipeline garantit

- **Pas de réindexation inutile** : `VectorDB` recharge la base persistée si
  elle existe déjà, et relit le nom du modèle d'embedding dans les métadonnées
  de la collection (pas dans `config.py`) pour empêcher tout désaccord
  silencieux entre le modèle utilisé à l'indexation et celui utilisé à la requête.
- **Le LLM principal n'est jamais appelé sur une tentative d'injection** :
  la modération se fait dans un appel séparé, avant tout le reste.
- **La base de connaissances fait foi** : si une question sort du corpus,
  le prompt impose de dire qu'on ne sait pas plutôt que d'inventer.
- **La recherche peut être restreinte par categorie** via `retrieve(question, categorie=...)`.

## Workflow git

Développement sur `dev`, une branche `feature/*` par brique, fusionnée avec
`--no-ff` pour garder la trace de chaque étape. `main` reçoit les versions
stabilisées de `dev`.
