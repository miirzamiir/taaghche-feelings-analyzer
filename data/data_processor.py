from typing import Tuple, List
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

"""
This class for crawl Taghche data for 
complecation dataset    name of the author,
                        name of the translator,
                        name of the book,
                        name of the publication
"""
class Crawler:
    
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
    
    def crawl(self, book_ids: list, book_names: str) -> List:
        author_list=[]
        translator_list=[]
        publisher_list=[]
        for index in tqdm(range(len(book_ids))):
            name = book_names[index].replace(" ", "_")
            id = str(int(book_ids[index]))
            url = self._base_url+"/"+id+"/"+name
            response = requests.get(url)
            if response.status_code == 200:
                page_content = response.content
                soup = BeautifulSoup(page_content, 'html.parser')
                book_data = soup.find_all(class_ = "bookHeader_detail__JY4wO")[0].get_text("|").split("|")
                author_list.append(book_data[book_data.index("نویسنده:")+1]
                                   if "نویسنده:" in book_data else "")
                translator_list.append(book_data[book_data.index("مترجم:")+1] 
                                       if "مترجم:" in book_data else "")
                publisher_list.append(book_data[book_data.index("انتشارات:")+1]
                                      if "انتشارات:" in book_data else "")
            else:
                print("Failed to retrieve the webpage url: "+ url)
        return [author_list, translator_list, publisher_list]
        
        