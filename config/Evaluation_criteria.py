import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from hazm import Normalizer, word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt


def simple_work(csv_file):

    # بارگذاری داده‌ها از فایل CSV
    # فرض کنیم داده‌ها در فایل csv_file ذخیره شده باشند
    df = pd.read_csv(csv_file)
    df['category'] = df['rating'].apply(categorize_review)

    # تعادل‌سازی داده‌ها
    positive_reviews = df[df['category'] == 'positive']
    neutral_reviews = df[df['category'] == 'neutral']
    negative_reviews = df[df['category'] == 'negative']

    # یافتن کمترین تعداد نمونه در بین دسته‌ها
    min_count = min(len(positive_reviews), len(neutral_reviews), len(negative_reviews))

    # نمونه‌برداری مجدد برای متعادل‌سازی داده‌ها
    positive_reviews_balanced = resample(positive_reviews, n_samples=min_count, random_state=42)
    neutral_reviews_balanced = resample(neutral_reviews, n_samples=min_count, random_state=42)
    negative_reviews_balanced = resample(negative_reviews, n_samples=min_count, random_state=42)

    # ترکیب داده‌های متعادل‌شده
    df_balanced = pd.concat([positive_reviews_balanced, neutral_reviews_balanced, negative_reviews_balanced])
    # نرمال‌سازی و توکن‌سازی متون فارسی
    normalizer = Normalizer()
    df_balanced['normalized_text'] = df_balanced['review'].apply(lambda text: ' '.join(word_tokenize(normalizer.normalize(text))))

    # تقسیم داده‌ها به مجموعه‌های آموزشی و تست
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['normalized_text'], df_balanced['category'], test_size=0.3, random_state=42)

    # تبدیل متون به ویژگی‌های TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # آموزش مدل Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # پیش‌بینی برچسب‌ها برای داده‌های تست
    y_pred = model.predict(X_test_tfidf)
    return y_test , y_pred


# تقسیم‌بندی نظرات به مثبت، خنثی و منفی
def categorize_review(rating):
    # تعریف ابرپارامترها برای تقسیم‌بندی امتیازات
    positive_threshold = 4
    negative_threshold = 2

    if rating > positive_threshold:
        return 'positive'
    elif rating < negative_threshold:
        return 'negative'
    else:
        return 'neutral'


def f1_score():
    y_test , y_pred = simple_work()
    # محاسبه F1-macro و F1-micro
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')

    print(f"F1-macro: {f1_macro}")
    print(f"F1-micro: {f1_micro}")


def accuracy_score():
    y_test , y_pred = simple_work()
    # محاسبه accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

def precision():
    y_test , y_pred = simple_work()
    # محاسبه precision برای هر دسته
    precision_positive = precision_score(y_test, y_pred, pos_label='positive', average='binary')
    precision_neutral = precision_score(y_test, y_pred, pos_label='neutral', average='binary')
    precision_negative = precision_score(y_test, y_pred, pos_label='negative', average='binary')

    print(f"Precision (positive): {precision_positive}")
    print(f"Precision (neutral): {precision_neutral}")
    print(f"Precision (negative): {precision_negative}")

    # محاسبه precision به صورت macro و micro
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_micro = precision_score(y_test, y_pred, average='micro')

    print(f"Precision (macro): {precision_macro}")
    print(f"Precision (micro): {precision_micro}")


def recall():
    y_test , y_pred = simple_work()
    # محاسبه recall برای هر دسته
    recall_positive = recall_score(y_test, y_pred, pos_label='positive', average='binary')
    recall_neutral = recall_score(y_test, y_pred, pos_label='neutral', average='binary')
    recall_negative = recall_score(y_test, y_pred, pos_label='negative', average='binary')

    print(f"Recall (positive): {recall_positive}")
    print(f"Recall (neutral): {recall_neutral}")
    print(f"Recall (negative): {recall_negative}")

    # محاسبه recall به صورت macro و micro
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')

    print(f"Recall (macro): {recall_macro}")
    print(f"Recall (micro): {recall_micro}")

def confusionMatrix():
    y_test , y_pred = simple_work()
    # محاسبه و نمایش ماتریس اغتشاش
    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])

    # ترسیم ماتریس اغتشاش
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
