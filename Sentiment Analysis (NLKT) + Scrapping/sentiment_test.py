import re

# import webdriver
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
# import Action chains 
from selenium. webdriver. common. keys import Keys

from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_comments():
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    driver = webdriver.Chrome(executable_path=r'C:\Users\Hammad\.wdm\drivers\chromedriver\win32\104.0.5112.79\chromedriver.exe') # to open the chromebrowser 
    driver.get('https://www.youtube.com/watch?v=4PydKcWm57w&ab_channel=SochTheBand')
    driver.maximize_window()
    time.sleep(3)
    elem = driver.find_element_by_tag_name("body")
    comments_data = []
    # count = scroll
    ActionChains(driver).move_to_element(elem)
    # check = 5
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

        # Wait to load page
        time.sleep(5)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.documentElement.scrollHeight")

        if new_height == last_height:
            break
        last_height = new_height    

    try:
        import pdb; pdb.set_trace()
        WebDriverWait(driver, 2).until(EC.presence_of_all_elements_located((By.XPATH, "//*[@id='contents']/ytd-comment-thread-renderer")))
        html_text = driver.page_source
        soup = BeautifulSoup(html_text, 'html.parser')
        no_comments = soup.find("h2", id="count").text
        no = no_comments.split('Comments')
        no = no[0].replace('\n', '')
        no = no.replace(' ', '')
        no = no.replace(',', '')
        no_comments = int(no)
        print(len(no_comments))
        try:
            while len(comments_data) < no_comments:
                comments = WebDriverWait(driver, 2).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//*[@id='contents']/ytd-comment-thread-renderer")))
                html_text = driver.page_source
                soup = BeautifulSoup(html_text, 'html.parser')

                for comment in comments:
                    html_text = comment.get_attribute("innerHTML")
                    soup = BeautifulSoup(html_text, 'html.parser')
                    text = soup.find("a", class_="yt-simple-endpoint style-scope yt-formatted-string").text
                    comment_ = soup.find(id="content-text").text
                    comments_data.append(comment_)

                elem.send_keys(Keys.PAGE_DOWN)
                checker = len(comments_data)


            return comments_data, no_comments
        except Exception as e:
            print(e)
            print("In Error section...")
            return [], ""
    except:
        print("Error.....")
        return [], ""


if __name__=='__main__':
    data,number=get_comments()
    print(number)