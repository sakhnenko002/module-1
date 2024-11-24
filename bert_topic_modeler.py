import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spacy

class BERTTopicModeler:
    def __init__(self, num_topics=10):
        """
        Ініціалізація моделі BERT для виявлення тем.
        
        :param num_topics: Кількість тем для кластеризації.
        """
        self.num_topics = num_topics
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.kmeans = None
        self.topic_labels = {}
        self.nlp = spacy.load("en_core_web_sm")  

    def encode_texts(self, texts):
        """
        Генерує векторні уявлення для текстів за допомогою BERT.
        
        :param texts: Список текстів (документів).
        :return: Матриця векторів для текстів.
        """
        embeddings = []
        for text in texts:
            
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def fit_cluster_model(self, embeddings, texts):
        """
        Навчає модель KMeans на векторах текстів.
        
        :param embeddings: Матриця векторів текстів.
        :param texts: Список текстів, що використовуються для створення міток тем.
        """
        self.kmeans = KMeans(n_clusters=self.num_topics, random_state=42)
        self.kmeans.fit(embeddings)

        
        cluster_centroids = self.kmeans.cluster_centers_
        self.topic_labels = {}

        for i, centroid in enumerate(cluster_centroids):
            similarities = cosine_similarity([centroid], embeddings)
            sorted_idx = similarities.argsort()[0][::-1]  

            
            top_doc = texts[sorted_idx[0]]
            
            self.topic_labels[i] = self.extract_main_concept(top_doc)

    def extract_main_concept(self, text):
        """
        Виділяє основне слово або концепцію з тексту, що представляє тему.
        
        :param text: Текст для аналізу.
        :return: Ключове слово або концепція.
        """
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'EVENT', 'PRODUCT']]
        
        if not entities:
            entities = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        
        main_concept = Counter(entities).most_common(1)
        return main_concept[0][0] if main_concept else "Unknown concept"

    def predict_topic(self, text):
        """
        Визначає тему для нового тексту.
        
        :param text: Текст для аналізу.
        :return: Назва теми.
        """
        if not self.kmeans:
            raise ValueError("Кластерна модель ще не навчена. Використовуйте fit_cluster_model спочатку.")
        
        
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)
        
        
        cluster_id = self.kmeans.predict(embedding)[0]
        return self.topic_labels.get(cluster_id, "Unknown topic")
