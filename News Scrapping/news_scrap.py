import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import threading


def check_title(dupli_check_data):
    global df
    for value in df['Title']:
        if dupli_check_data == value:
            return True
    return False


def data_csv(title, description, date, url, ):
    global df
    if not check_title(title):
        # update pandas dataframe
        df = df.append({'Title': title, 'Description': description, 'Date': date, 'URL': url}, ignore_index=True)
        
        # Export to CSV
        df.to_csv('news_data.csv', index=False)
        print("Data Saved :: Title: ", str(df['Title']))


def get_description_and_url(url, title, articles_list):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    description = soup.find('div', class_='story-area')

    news_description = ""
    news_date = ""

    if description:
    # Find all paragraphs within the story area
        paragraphs = description.find_all("p")
        
        # Extract text from each paragraph
        for paragraph in paragraphs:
            if re.findall(days_pattern, paragraph.text.strip()) and re.findall(month_pattern, paragraph.text.strip()):
                news_date = paragraph.text.strip()
            else:
                if news_description == "":
                    news_description = paragraph.get_text()
                else:
                    news_description = news_description + " " + paragraph.get_text()

        # Export Data
        data_csv(title, news_description, news_date, url)
    
    
    other_news = soup.find('div', class_='more-list row')

    if other_news:
        divs_inside_other_news = other_news.find_all('div')
        for article in divs_inside_other_news:
            articles_list.append(article)


def export_data(title, article, articles_list):
    try:
        news_url = article.find('a')['href']
        get_description_and_url(news_url, title, articles_list)
    except Exception as e:
        print(e)
        print("Didn't find any url")
        pass        


def extract_info(articles_list):
    while True:
        if articles_list:
            article = articles_list.pop(0)
            title = article.text.strip()
            if title:
                export_data(title, article, articles_list)


def scrape_news(base_url, articles_list):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')

    for article in articles:
        articles_list.append(article)


# # def clean_description(soup):
# #     news_description = soup.find('meta', {'name': 'description'}).get('content')


def main():
    global df
    # multiprocessing.freeze_support()
    # manager = Manager()
    # articles_list = manager.list()

    articles_list = []
    df = pd.DataFrame(columns=['Title', 'Description', 'Date', 'URL'])

    # p = Process(target=extract_info, args=(articles_list, df))
    # p.start()

    t = threading.Thread(target = extract_info, args = (articles_list,))
    t.start()

    news_url = 'https://www.geo.tv/'  # Replace this with the actual URL where news articles are listed
    scrape_news(news_url, articles_list)
    # p.join()


if __name__ == "__main__":
    days_pattern = r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
    month_pattern = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"

    main()
