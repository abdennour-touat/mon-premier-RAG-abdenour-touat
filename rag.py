"""
Brique 3 : le RAG qui orchestre tout.

Pipeline de answer_question() :
  1. modération de la question -> si injection détectée, on s'arrête là et
     on ne contacte JAMAIS le LLM principal (décision de sécurité : l'ordre
     des opérations compte).
  2. récupération des chunks les plus proches dans la base vectorielle.
  3. construction du prompt système à trous (remplacement de {{Chunks}}).
  4. appel au LLM Groq avec system + user.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from config import CORPUS_CSV_PATH, LLM_MODEL_NAME, N_RESULTS
from moderator import Moderator
from vector_db import VectorDB

RAG_PROMPT_PATH = Path(__file__).parent / "prompts" / "rag_system_prompt.txt"

REFUSAL_MESSAGE = (
    "Je ne peux pas traiter cette question : elle ressemble à une tentative "
    "de contourner mes consignes."
)


class RAG:
    def __init__(self, vector_db: VectorDB | None = None):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY manquante : ajoute-la dans un fichier .env "
                "(voir .env.example)."
            )

        self.client = Groq(api_key=api_key)
        self.moderator = Moderator(self.client)
        self.vector_db = vector_db or VectorDB(csv_path=CORPUS_CSV_PATH)
        self.system_prompt_template = RAG_PROMPT_PATH.read_text(encoding="utf-8")

    def _build_system_prompt(self, question: str) -> str:
        chunks = self.vector_db.retrieve(question, n=N_RESULTS)
        formatted_chunks = "\n".join(f"- {c['text']}" for c in chunks)
        return self.system_prompt_template.replace("{{Chunks}}", formatted_chunks)

    def answer_question(self, question: str) -> str:
        moderation = self.moderator.moderate(question)
        if moderation.get("is_prompt_injection"):
            return REFUSAL_MESSAGE

        system_prompt = self._build_system_prompt(question)

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
