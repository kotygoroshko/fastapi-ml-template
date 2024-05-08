from fastapi import APIRouter
from pydantic import BaseModel
from app.api.model import  Model 

inference_router = APIRouter()

class BodyText(BaseModel):
    text: str

model = Model()

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
    if model.model_train_raning:
        return {'warning': 'model in training mode! Wait!'}             
    else:
        if model.model_readynes:
            return {'warnig': 'model already trained!!!'}
        else:
            model.train_model()
            return model.metrics()
        
        
