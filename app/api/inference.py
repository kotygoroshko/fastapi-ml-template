from fastapi import APIRouter
from pydantic import BaseModel
from app.api.model import  Model 

inference_router = APIRouter()

class BodyText(BaseModel):
    text: str

model = Model()

@inference_router.get('/load_model/') 
def load_model()-> dict:
    model.load_model() 
    return {"load_model": "OK"}

@inference_router.post('/inference/')
def inference(body_text: BodyText) -> dict:
    """
    inference input text
    """ 
    return model.predict(body_text.text)

@inference_router.get('/train/')
def train() -> dict:
    """
    train
    """ 
    model.train_model()
    return model.metrics()