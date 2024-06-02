import re
import string
import threading

import spacy 
from nltk.corpus import brown


from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
import numpy as np

class SpaCySingleton:
    _model_instance = None
    _model_readynes = False
    _model_train_raning = False
    _lock = threading.Lock()
    _mode_name = "en_core_web_sm"

    @staticmethod
    def load_model():
        # Load the model
        try:
            SpaCySingleton._model_instance = spacy.load(SpaCySingleton._mode_name)
            SpaCySingleton._model_readynes = True
        except FileNotFoundError:
            SpaCySingleton._model_readynes = False
            SpaCySingleton._model_instance = None
            print(f"The {SpaCySingleton._mode_name} does not exist.")
        except Exception as e:
            SpaCySingleton._model_readynes = False
            print(f"An error occurred: {e}")

    @staticmethod
    def get_instance():
        if SpaCySingleton._model_instance is None:
            with SpaCySingleton._lock:
                if SpaCySingleton._model_instance is None:
                    SpaCySingleton.load_model()
        return SpaCySingleton._model_instance


    @staticmethod
    def pre_procesing(sents):
        def is_token_allowed(token):
            return bool(
                token
                and str(token).strip()
                and not token.is_stop
                and token.is_alpha
                )  

        def preprocess_token(token):
            return token.lemma_.strip().lower()

        doc = SpaCySingleton.get_instance()(sents)

        complete_filtered_tokens = [
            preprocess_token(token)
            for token in doc
            if is_token_allowed(token)
        ]

        return complete_filtered_tokens
