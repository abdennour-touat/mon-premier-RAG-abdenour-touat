"""
Brique 2 : l'agent modérateur.

Confié à un modèle dédié ("safeguard") plutôt que dilué dans le prompt du
RAG lui-même, pour deux raisons :
  1. séparation des responsabilités : un modèle spécialisé en classification
     de sécurité est entraîné et évalué spécifiquement pour ça, plus fiable
     qu'une instruction noyée parmi les consignes métier du RAG ;
  2. l'ordre des opérations devient une garantie testable : on peut vérifier
     que le LLM principal n'est JAMAIS appelé si is_prompt_injection=True,
     ce qui serait impossible à garantir si la détection et la génération
     étaient mélangées dans un seul appel.
"""

import json
from pathlib import Path

from groq import Groq

from config import MODERATION_MODEL_NAME

PROMPT_PATH = Path(__file__).parent / "prompts" / "moderator_system_prompt.txt"


class Moderator:
    def __init__(self, client: Groq):
        self.client = client
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def moderate(self, question: str) -> dict:
        response = self.client.chat.completions.create(
            model=MODERATION_MODEL_NAME,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
