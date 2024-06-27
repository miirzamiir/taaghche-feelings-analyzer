import os
from data_processor import *
import pandas as pd

# data = pd.read_csv("taghche.csv")

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "taghche.csv")
data = pd.read_csv(csv_path)

data = data.dropna(subset=['bookID'])

taghche_url = "https://taaghche.com/book"
crawler = Crawler(taghche_url)

book_ids = list(set(data.bookID.tolist()))

for i, id in enumerate(book_ids):
    if not id:
        book_ids.pop(i)

i = 10600
j = 54
while i<len(book_ids):
    print(f'======== BATCH {j} ========\n')
    lower = i
    upper = min(i+200, len(book_ids))
    result = crawler.crawl(book_ids[lower:upper])
    df = pd.DataFrame(result, columns=['book id', 'author', 'translator', 'publisher'])
    df.to_csv('data/atp.csv', mode='a', index=False)

    i += 200
    j += 1

