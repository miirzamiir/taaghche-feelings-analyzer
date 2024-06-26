import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import *
from tensorflow_addons.layers import *

data = pd.DataFrame({
    'review': ["متن نظر ۱", "متن نظر ۲", "متن نظر ۳"],  # نمونه‌های فرضی
    'rating': [5, 3, 1]  # نمونه‌های فرضی
})

# تبدیل امتیازات به سه دسته (مثبت، منفی، خنثی)
def sentiment_label(rating):
    if rating >= 4:
        return 'positive'
    elif rating >= 2:
        return 'neutral'
    else:
        return 'negative'

data['sentiment'] = data['rating'].apply(sentiment_label)

# تبدیل برچسب‌ها به اعداد
label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
data['label'] = data['sentiment'].map(label_mapping)


def char_embeddings(text, vocab, max_length):
    return [vocab.get(char, 0) for char in text[:max_length]]

def segmentation_info(text, max_length):
    return [1 if char == ' ' else 0 for char in text[:max_length]]

vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")}

# پارامترها
max_length = 100
character_vocab_size = len(vocab) + 1
segmentation_vocab_size = 2
embedding_dim = 64
lstm_units = 128
num_labels = len(label_mapping)

def lstm_fun(data, vocab, max_length, num_layers):

    data['char_emb'] = data['review'].apply(lambda x: char_embeddings(x, vocab, max_length))
    data['seg_info'] = data['review'].apply(lambda x: segmentation_info(x, max_length))

    X_char = np.array(data['char_emb'].tolist())
    X_seg = np.array(data['seg_info'].tolist())
    y = np.array(data['label'].tolist())

    # ورودی‌های لایه
    character_input = Input(shape=(max_length,), dtype='int32', name='character_input')
    segmentation_input = Input(shape=(max_length,), dtype='int32', name='segmentation_input')

    # تعریف لایه‌های LSTM بر اساس تعداد لایه‌ها
    lstm_layers = []
    for _ in range(num_layers):
        lstm_layer = LSTM(units=lstm_units, return_sequences=True)
        lstm_layers.append(lstm_layer)

    character_embedding = Embedding(input_dim=character_vocab_size, output_dim=embedding_dim, mask_zero=True)(character_input)
    # segmentation_embedding = Embedding(input_dim=segmentation_vocab_size, output_dim=embedding_dim, mask_zero=True)(segmentation_input)

    # اعمال لایه‌های LSTM
    lstm_outputs = [lstm_layer(character_embedding if i == 0 else lstm_outputs[i-1]) for i, lstm_layer in enumerate(lstm_layers)]

    combined = concatenate(lstm_outputs, axis=-1)

    dense = TimeDistributed(Dense(num_labels))(combined)
    crf = CRF(num_labels)
    output = crf(dense)

    model = Model(inputs=[character_input, segmentation_input], outputs=output)

    try:
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
    except AttributeError:
        pass

    model.summary()

    model.fit([X_char, X_seg], y, batch_size=32, epochs=10)

