from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from app.api.spacy import SpaCySingleton

spa_cy_router = APIRouter()

class SentencesRequest(BaseModel):
    sentences: List[str]

@spa_cy_router.post("/spacy/pre_procesing")
def pre_procesing(text: SentencesRequest):
    SpaCySingleton.get_instance()
    if SpaCySingleton._model_readynes:
        return {i: SpaCySingleton.pre_procesing(sentence)  for i,sentence in enumerate(text.sentences)}
    else:
        return {'error':'Model not ready!!!'}
