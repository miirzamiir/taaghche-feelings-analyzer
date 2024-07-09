import re
import cupy as cp
from hazm import *
import pandas as pd
from cleantext import clean
from farsi_tools import stop_words
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
        with open('resources/stopwords.txt',encoding="utf-8") as f:
            stop=f.readlines()
        stop_word=[word.replace('\n','') for word in stop]
        stop_word=[re.sub('[\\u200c]',' ',word)  for word in stop_word]
        stop_word.extend(stop_words())

    def clean_text(self,text):
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
        text = re.sub(r'[\u200c]', '', text)
        text=re.sub(r'[A-Za-z0-9]','',text)
        text=re.sub(r'[^\w\s]', '', text)
        text=re.sub(r'[۔،؛؟٪٬…_]', '', text)
        text=re.sub(r'[\n]','',text)
        text=re.sub(' +',' ',text)
        text = word_tokenize(text)
        text=[word for word in text if word not in self.stop_word]
        text=' '.join(text)

        return text

    def final_clean(self, data):
        data = data.drop_duplicates()
        clean_dataset = data.dropna(subset=['rate'])
        clean_dataset['clean'] = clean_dataset['comment'].apply(self.clean_text)
        dd=[3.0, 4.0, 5.0]
        filtered_df = clean_dataset[clean_dataset['rate'].isin(dd)]
        for i in dd:
            filtered_f=filtered_df[filtered_df['rate']==i]
            rem=filtered_f.iloc[:100]
            filtered_df=pd.concat([filtered_df[filtered_df['rate']!=i],rem])
        return filtered_df

    def preprocess(self, text):
        text = self.normalizer.normalize(text)
        text = self.final_clean(text)
        tokens = word_tokenize(text)
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
        x_train_gpu = cp.asarray(x_train)
        y_train_gpu = cp.asarray(y_train)
        x_test_gpu = cp.asarray(x_test)

        # Initialize and train the model on GPU
        svm_model = SVC(kernel='poly', degree=3, C=1)
        svm_model.fit(x_train_gpu, y_train_gpu)

        # Make predictions on the GPU
        y_pred_svm_gpu = svm_model.predict(x_test_gpu)

        # Convert predictions back to NumPy arrays for compatibility with scikit-learn metrics
        y_pred_svm = cp.asnumpy(y_pred_svm_gpu)

        # Print results
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