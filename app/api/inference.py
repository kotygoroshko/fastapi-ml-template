from fastapi import APIRouter
from pydantic import BaseModel
from app.api.model import  Model
from typing import List 

inference_router = APIRouter()


model = Model()

class ItemRequest(BaseModel):
  text: List[str]

@inference_router.post('/inference/')
def inference(request: ItemRequest) -> dict:
    """
    inference input text
    """ 
    return model.predict(request.text)

@inference_router.get('/train/')
def train() -> dict:
    """
    train
    """ 
    if model.model_train_raning:
        return {'warning': 'model in training mode! Wait!'}             
    else:
        if model.model_readynes:
            return {'warnig': 'model already trained!!!'}
        else:
            model.train_model()
            return model.metrics()
        
        
