"""
Brique 1 : la base vectorielle persistante.

VectorDB se comporte différemment selon la situation :
- si une base existe déjà sur disque -> elle la recharge ;
- sinon, si on lui fournit un chemin vers un CSV -> elle la crée à partir
  de ce CSV (colonnes attendues : id, text, source, categorie) ;
- sinon -> erreur explicite.

Astuce reprise de la démo : le nom du modèle d'embedding est stocké dans les
métadonnées de la collection elle-même. Au rechargement, on relit CE nom-là,
et pas celui (peut-être différent) écrit dans config.py. Ça empêche un bug
silencieux : si on changeait EMBEDDING_MODEL_NAME dans config.py après avoir
indexé une base, interroger cette base avec un nouveau modèle produirait des
vecteurs de dimension/sémantique différente de ceux stockés -> des résultats
de similarité incohérents, sans aucune erreur pour te prévenir.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME
from data.load_corpus import load_corpus


class VectorDB:
    def __init__(self, path: str = CHROMA_DB_PATH, csv_path: str | None = None):
        self.path = path
        self.client = chromadb.PersistentClient(path=self.path)

        collection_exists = COLLECTION_NAME in [
            c.name for c in self.client.list_collections()
        ]

        if collection_exists:
            self._load_existing()
        elif csv_path:
            self._create_from_csv(csv_path)
        else:
            raise ValueError(
                f"Aucune base trouvée dans '{self.path}' et aucun `csv_path` "
                "fourni pour en créer une."
            )

    def _create_from_csv(self, csv_path: str) -> None:
        records = load_corpus(csv_path)

        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Le nom du modèle est écrit dans les métadonnées de la collection,
        # pas seulement dans config.py.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": self.embedding_model_name},
        )

        texts = [r["text"] for r in records]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # cosine similarity <=> produit scalaire
            show_progress_bar=False,
        )

        self.collection.add(
            ids=[r["id"] for r in records],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"source": r["source"], "categorie": r["categorie"]}
                for r in records
            ],
        )

    def _load_existing(self) -> None:
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

        # On relit le nom du modèle dans les métadonnées de la collection,
        # pas dans config.py : c'est ce qui garantit que la requête est
        # toujours encodée avec le même modèle que l'indexation, même si
        # config.py a changé entre-temps.
        self.embedding_model_name = self.collection.metadata["embedding_model"]
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def retrieve(
        self, question: str, n: int = 3, categorie: str | None = None
    ) -> list[dict]:
        """Retourne les n chunks les plus proches de la question, triés du
        plus au moins pertinent, avec leur texte, leur source et leur
        catégorie.

        Si `categorie` est fourni, la recherche est restreinte aux chunks
        de cette catégorie (utile par exemple pour ne chercher que parmi
        les faits classés "animaux")."""
        query_embedding = self.embedding_model.encode(
            [question], normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n,
            where={"categorie": categorie} if categorie else None,
        )

        return [
            {
                "text": doc,
                "source": meta["source"],
                "categorie": meta["categorie"],
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
