from selenium.common import TimeoutException
# For data classification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from selenium.webdriver.common.by import By
import time

# For data scraping
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


#STEP 1: get all info from my 10 pro seed videos, 10 con seed videos



# Function to scrape video details, returns a pd dataframe
def scrape_video_details(video_links, driver):
    titles = []
    descriptions = []
    labels = []
    likes = []
    ids = []

    df = pd.DataFrame(columns = ['id', 'link', 'title', 'description', 'category'])
    wait = WebDriverWait(driver, 10)
    v_category = "CATEGORY_NAME"

    for link in video_links:
        driver.get(link)
        v_id = link.strip('https://www.youtube.com/watch?v=')
        v_title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
        v_description = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#descriptionyt-formatted-string"))).text
        df.loc[len(df)] = [v_id, v_title, v_description, v_category]

        ids.append(v_id)
        titles.append(v_title)
        descriptions.append(v_description)
        #labels.append(label)
        #likes.append(like_count)

    driver.quit()
    return titles, descriptions, labels, likes, ids

# Function to read video links from a text file
def read_video_links_from_file(file_path):
    with open(file_path, 'r') as file:
        video_links = [line.strip() for line in file.readlines()]
    return video_links

#Create a chrome driver
def create_chrome_driver(path):
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-web-security")
    options.add_argument('--user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"')
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--lang=en-US")
    """
    #comment out in order to see the scraper interacting with webpages
    #options.add_argument('--headless')
    #find a way to fix options
    return webdriver.Chrome()



if __name__ == "__main__":

    # init the webdriver
    driver = create_chrome_driver('~/Desktop/COGS401/Scraper_data')

    # seed search "high restriction what i eat in a day"
    v1 = "https://www.youtube.com/results?search_query=wieiad+ed+high+res"
    driver.get(v1)

    # get all search results
    user_data = driver.find_elements(By.XPATH,'//*[@id="video-title"]')
    links = []
    for i in user_data:
        links.append(i.get_attribute('href'))
    print(links)

    # create dataframe to store all attributes
    """
    df = pd.DataFrame(columns=['link', 'title', 'description', 'category'])
    wait = WebDriverWait(driver, 10)
    v_category = "pro_ed_search1"
    for x in links:
        try:
            driver.get(x)
            v_id = x.strip('https://www.youtube.com/watch?v=')
            v_title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
            v_description =  wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#descriptionyt-formatted-string"))).text
            df.loc[len(df)] = [v_id, v_title, v_description, v_category]
        except TimeoutException:
            print("Element did not show up")

    """
    # Read pro-ed and con-ed video links from text files
    #pro_ed_links = read_video_links_from_file('seeds_pro_ed.txt')
    #con_ed_links = read_video_links_from_file('seeds_con_ed.txt')

    # Scrape details for pro-ed and con-ed videos
    #pro_titles, pro_descriptions, pro_labels, pro_likes = scrape_video_details(pro_ed_links, scraper)
    #con_titles, con_descriptions, con_labels, con_likes = scrape_video_details(con_ed_links, scraper)

    # Now you have separate lists for pro and con video details
    #print(pro_titles)

    # close webdriver
    driver.quit()