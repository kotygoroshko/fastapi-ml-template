import re
import string
import threading

from nltk.corpus import brown
from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
import numpy as np

class Doc2VecSingleton:
    _model_instance = None
    _model_readynes = False
    _model_train_raning = False
    _lock = threading.Lock()
    _mode_file_name = "hw15_doc2vec_model.bin"

    @staticmethod
    def load_model():
        # Load the model
        try:
            Doc2VecSingleton._model_instance = Doc2Vec.load(Doc2VecSingleton._mode_file_name)
            Doc2VecSingleton._model_readynes = True
        except FileNotFoundError:
            Doc2VecSingleton._model_readynes = False
            Doc2VecSingleton._model_instance = Doc2Vec(vector_size=40, min_count=2, epochs=30)
            print(f"The file {Doc2VecSingleton._mode_file_name} does not exist.")
        except Exception as e:
            Doc2VecSingleton._model_readynes = False
            print(f"An error occurred: {e}")

    @staticmethod
    def get_instance() -> Doc2Vec:
        if Doc2VecSingleton._model_instance is None:
            with Doc2VecSingleton._lock:
                if Doc2VecSingleton._model_instance is None:
                    Doc2VecSingleton.load_model()
        return Doc2VecSingleton._model_instance
    

    @staticmethod
    def train():
        Doc2VecSingleton._model_train_raning = True
        sents = brown.sents()
        sents = Doc2VecSingleton.pre_procesing(sents)
        data_for_training = list(Doc2VecSingleton.tagged_document(sents))
        Doc2VecSingleton.get_instance().build_vocab(data_for_training)
        Doc2VecSingleton.get_instance().train(data_for_training, 
                                               total_examples=Doc2VecSingleton._model_instance.corpus_count, 
                                               epochs=Doc2VecSingleton._model_instance.epochs)
        Doc2VecSingleton.get_instance().save("hw15_doc2vec_model.bin")
        Doc2VecSingleton._model_readynes = True
        Doc2VecSingleton._model_train_raning = False


    @staticmethod
    def pre_procesing(sents):
        # write the removal characters such as : Stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—'
        string.punctuation
        removal_list = list(stop_words) + list(string.punctuation)+ ['lt','rt'] + ['``',"''"]
        
        # Remove Emails
        sents = [[re.sub('\S*@\S*\s?', '', word) for word in sent] for sent in sents ]

        # Remove removal_list
        sents = [[word for word in sent if word not in removal_list] for sent in sents ]

        return sents
    
    @staticmethod
    def tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield TaggedDocument(list_of_words, [i])

    @staticmethod    
    def group_sentences(sentences):
        sentences_words = [re.findall(r"\w+", sentence) for sentence in sentences]
        print(sentences_words)
        sentence_vectors = Doc2VecSingleton.get_sentence_vectors(sentences_words)

        # Виконання ієрархічної кластеризації з косинусною відстанню
        threshold = 0.5  # Порогова відстань для визначення кластерів
        labels = Doc2VecSingleton.hierarchical_clustering(sentence_vectors, threshold)
        grouped_sentences = Doc2VecSingleton.group_sentences_by_theme(sentences, labels)
        print(f'grouped_sentences {grouped_sentences}')
        return grouped_sentences

    # Отримання векторів речень
    @staticmethod
    def get_sentence_vectors(sentences):
        return [Doc2VecSingleton.get_instance().infer_vector(doc_words=sentence) for sentence in Doc2VecSingleton.pre_procesing(sentences)]

    # Ієрархічна кластеризація з використанням косинусної відстані
    def hierarchical_clustering(vectors, threshold=0.5):
        # Обчислення матриці попарних косинусних відстаней
        distance_matrix = np.array([[cosine(v1, v2) for v2 in vectors] for v1 in vectors])
        print(f'distance_matrix : {distance_matrix}')
        # Виконання ієрархічної кластеризації
        Z = linkage(distance_matrix, 'complete')
        
        # Фіксування кластерів за заданим порогом
        labels = fcluster(Z, t=threshold, criterion='distance')
        return labels.astype(str)

    # Групування речень по темах
    def group_sentences_by_theme(sentences, labels):
        grouped_sentences = {}
        for label, sentence in zip(labels, sentences):
            if label not in grouped_sentences:
                grouped_sentences[label] = []
            grouped_sentences[label].append(sentence)
        return grouped_sentences