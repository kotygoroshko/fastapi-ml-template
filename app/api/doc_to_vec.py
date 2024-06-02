from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from app.api.doc2vec import Doc2VecSingleton
import re

doc_to_vec_router = APIRouter()

class SentencesRequest(BaseModel):
    sentences: List[str]

@doc_to_vec_router.post("/doc2vec/classify_sentences")
def classify_sentences(text: SentencesRequest):
    Doc2VecSingleton.get_instance()
    if Doc2VecSingleton._model_readynes:
        return Doc2VecSingleton.group_sentences(text.sentences)
    else:
        return {'error':'Model not ready!!!'}

@doc_to_vec_router.get('/doc2vec/train/')
def train() -> dict:
    """
    train
    """ 
    if Doc2VecSingleton._model_train_raning:
        return {'warning': 'model in training mode! Wait!'}             
    else:
        if Doc2VecSingleton._model_readynes:
            return {'warnig': 'model already trained!!!'}
        else:
            Doc2VecSingleton.train()
            return {'responce': 'model trained!!!'}
    

