import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
import string

# Clean text
def clean_text(text):
    # Make lower
    text = text.lower()
    # Remove line breaks
    text = re.sub(r'\n', '', text)
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

async def load_post(url_post):
    # 0. Define browser
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Chạy không hiển thị trình duyệt
    chrome_options.add_argument("--disable-gpu")  # Tắt GPU để tăng hiệu năng
    browser = webdriver.Chrome(options=chrome_options)
    

    # 1. open Facebook
    browser.get("http://facebook.com")

    # 2.Load cookie from file

    cookies = pickle.load(open("./static/my_cookie.pkl","rb"))
    for cookie in cookies:
        browser.add_cookie(cookie)
    browser.get(url_post)
    text_post = browser.find_element(By.XPATH, "/html/body/div[1]/div/div/div[1]/div/div[5]/div/div/div[2]/div/div/div/div/div/div/div/div[2]/div[2]/div/div/div/div/div/div/div/div/div/div/div/div/div[13]/div/div/div[3]/div[1]")
    return clean_text(text_post.text)

