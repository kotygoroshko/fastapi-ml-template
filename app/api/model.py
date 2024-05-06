import pandas as pd

import zipfile
import string
import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib


class Model:
    
    def __init__(self):
        self.flags = {
            'remove_tags_flag': True,
            'remove_links_flag' : True,
            'remove_punctuation_flag' : True,
            'remove_small_words_flag': True,
        }

        self.model = LogisticRegression(multi_class= 'ovr' , random_state = 42, max_iter=1000)

    def load_data(self):
        # Открываем ZIP-архив
        with zipfile.ZipFile('data/archive.zip', 'r') as zip_file:
            # Извлекаем список файлов в архиве
            file_list = zip_file.namelist()

            # Читаем первый CSV-файл из архива в DataFrame
            with zip_file.open('IMDB Dataset.csv') as csv_file:
                df = pd.read_csv(csv_file, delimiter=',')
        return df



    def remove_links(self, text):
        """
        Removes URLs from a string using regular expressions.

        Args:
            text: The input string.

        Returns:
            A new string with URLs removed.
        """
        if isinstance(text, str):
            return re.sub(r'https?://\S+', '', text)

    def remove_punctuation(self, text):
        """
        Removes punctuation marks from a string.

        Args:
            text: The input string.

        Returns:
            A new string with punctuation marks removed.
        """
        if isinstance(text, str):
            pattern = f'[{string.punctuation}\d]'
            return re.sub(pattern, ' ',text)

    def remove_small_words(self, text, min_length=1):
        """
        Removes words from a string that are shorter than the specified minimum length.

        Args:
            text: The input string.
            min_length: The minimum length a word must have to be kept (default: 1).

        Returns:
            A new string with short words removed.
        """
        if isinstance(text, str):
            words = text.split()  # Split the string into words
            filtered_words = [word for word in words if len(word) > min_length]  # Filter words based on length
            return " ".join(filtered_words)  # Join the filtered words back into a string

    def remove_tags(self, text):
        """Removes tags from text based on a start/end tag format (e.g., <tag>text</tag>)."""
        pattern = r"<.*?>"  # Regular expression for tag removal
        return re.sub(pattern, '', text)  # Replace tags with their content
    
    def create_target(self):
        # Define a dictionary for mapping sentiment to numerical values
        sentiment_mapping = {'positive': 1, 'negative': 0}

        # Replace sentiment values in the 'sentiment' column using the mapping
        self.data['target'] = self.data['sentiment'].replace(sentiment_mapping)

    def preprocessing(self, text):
        self.create_target()
        if self.flags.get('remove_tags_flag', False):
            text = self.remove_tags(text)
        if self.flags.get('remove_links_flag', False):
            text = self.remove_links(text)
        if self.flags.get('remove_punctuation_flag', False):
            text = self.remove_punctuation(text)
        if self.flags.get('remove_small_words_flag', False):
            text = self.remove_small_words(text)
        return text

    def train_model(self):
        self.data = self.load_data()
        self.data['review'].apply(self.preprocessing)

        # Split data (80% for training, 20% for testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['review'], self.data['target'], test_size=0.2, random_state=42)

        # Робимо фічі за допомогою BoW
        vectorizer = CountVectorizer(stop_words='english', max_features=1_000)
        vectorizer.fit(self.X_train)
        # Продивляємось що попало в словарик.
        vocabulary = vectorizer.get_feature_names_out()
        print(vocabulary)
        print(len(vocabulary))
        # Трансформуємо X_train та X_test для роботи з моделью.
        self.X_train_features = vectorizer.transform(self.X_train)
        self.X_test_features = vectorizer.transform(self.X_test)
        # тренуємо модел 
        self.model.fit(self.X_train_features, self.y_train)

        # Дивимося метрики моделі
        self.metrics()


        # Save the model to a file
        joblib.dump(self.model, 'model.joblib')

    def metrics(self):
        # Дивимося метрики моделі
        y_pred = self.model.predict(self.X_test_features)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

        return {"Accuracy:": accuracy, "Precision:": precision, "Recall:": recall, "F1-Score:": f1}

    def load_model(self):
        # Later, to load the model
        self.model = joblib.load('model.joblib')
    

        