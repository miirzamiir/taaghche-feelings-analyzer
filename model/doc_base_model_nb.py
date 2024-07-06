from hazm import *
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


class DocBaseNiveBayes:
    def __init__(self) -> None:
        self.normalizer = Normalizer()
        self.stop_words = set(stopwords_list())
        self.tokenizer = WordTokenizer()

    def preprocess(self, text):
        text = self.normalizer.normalize(text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and word.isalpha()]
        return ' '.join(tokens)

    def tfidf_vectorizer(self, data):
        vectorizer = TfidfVectorizer()
        data['comment'] = data['comment'].astype(str).apply(self.preprocess)
        tfidf_matrix = vectorizer.fit_transform(data['comment'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        return tfidf_df

    def data_preparation(self, x, y):
        rus = RandomUnderSampler(random_state=0)
        x_under, y_under = rus.fit_resample(x, y)
        return x_under, y_under

    def naive_bayes_classifier(self, x_train, y_train, x_test, y_test):
        nb_model = MultinomialNB()
        nb_model.fit(x_train, y_train)
        y_pred_nb = nb_model.predict(x_test)
        print("Naive Bayes Classifier")
        print("Accuracy:", accuracy_score(y_test, y_pred_nb))
        print(classification_report(y_test, y_pred_nb))

    def svm_classifier(self, x_train, y_train, x_test, y_test):
        # svm_model = SVC(kernel='linear')
        # svm_model.fit(x_train, y_train)
        svm_model = SVC(kernel='poly', degree=3, C=1).fit(x_train, y_train)
        y_pred_svm = svm_model.predict(x_test)
        print("Support Vector Machine Classifier")
        print("Accuracy:", accuracy_score(y_test, y_pred_svm))
        print(classification_report(y_test, y_pred_svm))

    def logistic_regression_classifier(self, x_train, y_train, x_test, y_test):
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(x_train, y_train)
        y_pred_lr = lr_model.predict(x_test)
        print("Logistic Regression Classifier")
        print("Accuracy:", accuracy_score(y_test, y_pred_lr))
        print(classification_report(y_test, y_pred_lr))

    def doc_base_model(self, datapath='data/taghche.csv'):
        data = pd.read_csv(datapath)
        data['category'] = data['rate'].apply(lambda x: 'BAD' if x < 3 else('NEUTRAL' if x < 5 else 'GOOD'))
        
        x, y = self.data_preparation(x=data.loc[:, ['comment']], y=data.loc[:, ['category']])
        x = self.tfidf_vectorizer(data=x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=42)
        print(y.value_counts())
        # self.naive_bayes_classifier(x_train, y_train, x_test, y_test)
        self.svm_classifier(x_train, y_train.values.ravel(), x_test, y_test.values.ravel())
        # self.logistic_regression_classifier(x_train, y_train.values.ravel(), x_test, y_test.values.ravel())


dbnb = DocBaseNiveBayes()
dbnb.doc_base_model()