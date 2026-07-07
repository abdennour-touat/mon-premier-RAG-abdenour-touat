"""
Constantes du projet.

Un seul endroit pour les noms de modèles et les chemins : évite que
l'indexation et la génération divergent silencieusement à cause d'un
paramètre différent écrit dans deux fichiers différents.
"""

# Modèle d'embedding utilisé pour indexer et interroger la base vectorielle.
# Multilingue et léger : suffisant pour un corpus de démonstration.
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

# Modèle Groq utilisé pour la génération de la réponse finale.
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# Modèle Groq dédié à la modération (détection de prompt injection).
MODERATION_MODEL_NAME = "openai/gpt-oss-120b"

# Emplacement de la base ChromaDB persistée sur disque.
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "toy_facts"

# Emplacement du corpus source, au format CSV (id, text, source, categorie).
CORPUS_CSV_PATH = "data/corpus.csv"

# Nombre de chunks les plus proches remontés à chaque question.
N_RESULTS = 3
