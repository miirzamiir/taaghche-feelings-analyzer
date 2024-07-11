import re
import numpy as np
from hazm import *
import pandas as pd
from cleantext import clean
from farsi_tools import stop_words
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

class DocBaseModel:
    def __init__(self) -> None:
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
        self.stop_words = stopwords_list()
        with open('resources/stopwords.txt', encoding="utf-8") as f:
            stop = f.readlines()
        self.stop_words.extend([word.replace('\n', '') for word in stop])
        self.stop_words.extend(stop_words())

    def to_lower(self, match):
        return match.group(0).lower()

    def lowercase_english_words(self, text):
        pattern = re.compile(r'[a-zA-Z]+')
        result = pattern.sub(self.to_lower, text)
        return result

    def clean_text(self, text):
        text = clean(text,
                     fix_unicode=True,
                     to_ascii=False,
                     no_numbers=True,
                     no_emoji=True,
                     no_digits=True,
                     no_punct=True,
                     no_emails=True,
                     replace_with_phone_number='',
                     replace_with_number='',
                     replace_with_digit='',
                     replace_with_email='',
                     replace_with_currency_symbol='',
                     replace_with_punct='')
        text = self.lowercase_english_words(text)
        text = re.sub(r'[\u200c]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[۔،؛؟٪٬…_]', '', text)
        text = re.sub(r'[\n]', '', text)
        text = re.sub(' +', ' ', text)
        text = word_tokenize(text)
        text = [word for word in text if word not in self.stop_words]
        text = ' '.join(text)
        return text

    def preprocess(self, text):
        text = self.normalizer.normalize(text)
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        return ' '.join(tokens)

    def tfidf_vectorizer(self, data, vectorizer=None):
        if vectorizer is None:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            data['comment'] = data['comment'].astype(str).apply(self.preprocess)
            tfidf_matrix = vectorizer.fit_transform(data['comment'])
        else:
            data['comment'] = data['comment'].astype(str).apply(self.preprocess)
            tfidf_matrix = vectorizer.transform(data['comment'])
        return tfidf_matrix, vectorizer

    def data_preparation(self, x, y):
        smote = SMOTE(random_state=0)
        x_over, y_over = smote.fit_resample(x, y)
        return x_over, y_over

    def batch_generator(self, x, y, batch_size):
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            yield x[start:end], y[start:end]

    def incremental_learning(self, model, x_train, y_train, batch_size, n_epochs):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            for x_batch, y_batch in self.batch_generator(x_train, y_train, batch_size):
                model.partial_fit(x_batch, y_batch, classes=np.unique(y_train))
        return model

    def cross_validation_scores(self, model, x, y, model_name):
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro']
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, x, y, cv=skf, scoring='accuracy')
        print(f"{model_name} - Cross Validation Results:")
        print(f"Accuracy: {np.mean(results)}")
        print(f"Precision (Macro): {np.mean(cross_val_score(model, x, y, cv=skf, scoring='precision_macro'))}")
        print(f"Recall (Macro): {np.mean(cross_val_score(model, x, y, cv=skf, scoring='recall_macro'))}")
        print(f"F1 Score (Macro): {np.mean(cross_val_score(model, x, y, cv=skf, scoring='f1_macro'))}")
        print(f"F1 Score (Micro): {np.mean(cross_val_score(model, x, y, cv=skf, scoring='f1_micro'))}\n")

    def confusion_matrix_report(self, model, x_test, y_test, model_name):
        y_pred = model.predict(x_test)
        print(f"{model_name} - Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\n")
        return model

    def naive_bayes_classifier(self, x_train, y_train, x_test, y_test, batch_size, n_epochs):
        nb_model = MultinomialNB()
        nb_model = self.incremental_learning(nb_model, x_train, y_train, batch_size, n_epochs)
        self.cross_validation_scores(nb_model, x_train, y_train, "Naive Bayes Classifier")
        nb_model = self.confusion_matrix_report(nb_model, x_test, y_test, "Naive Bayes Classifier")
        joblib.dump(nb_model, 'model/naive_bayes_model.pkl')

    def doc_base_model(self, datapath='data/taghche.csv'):
        data = pd.read_csv(datapath)
        data = data.drop_duplicates()
        data = data.dropna(subset=['rate'])
        data['category'] = data['rate'].apply(lambda x: 'BAD' if x < 2 else ('NEUTRAL' if x < 5 else 'GOOD'))
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        data['comment'] = data['comment'].astype(str).apply(self.preprocess)
        tfidf_matrix = vectorizer.fit_transform(data['comment'])
        
        y = data['category']
        x, y = self.data_preparation(tfidf_matrix, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)
        
        print(y.value_counts())
        
        batch_size = 1000
        n_epochs = 5
        
        self.naive_bayes_classifier(x_train, y_train, x_test, y_test, batch_size, n_epochs)
        
dbm = DocBaseModel()
dbm.doc_base_model()
