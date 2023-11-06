from googleapiclient.discovery import build
import pandas as pd

#GLOBAL VARS
maxResults = 30 #want 40 videos per keyword, 10 on each page visited
category = [] # Data to be stored


# Gathering Data using the yt API, have many because daily quota easy to exceed
API_KEY1 = "AIzaSyAh1sVjvm7b7GH0_W-1AXYOFldVvgRM9Gk"
API_KEY2 = "AIzaSyAfIe8O946wN41Bla1JnRAGooASlNWMX1I"
API_KEY3 = "AIzaSyAUZXU9m9WdIxRFHAdK2lBbCtwgFQPO-tI"
API_KEY4 = "AIzaSyAWPuynMfLH1u0M_tOPytdqXHuLo80Yjeo"
API_KEY5 = "AIzaSyD5igxO2TntNPXzaWe2EljzbRzATx_vRf8"
API_KEY6 ="AIzaSyCBLJcm3WV7n-de3tns92yA9qTF8BZUvfQ"
API_KEY7 = "AIzaSyAS9eTgOEnOJ2GlJbbqm_0bR1onuRQjTHE"
API_KEY8 = "AIzaSyAg6dxTbCfUlEFyKffbdQ-JsrddL90uZus"

#Alternate keys to ensure we don't go over quota
youtube_api = build('youtube','v3', developerKey = API_KEY1)

#Reads all keywords from label file
def read_keywords_from_file(file_path):
    with open(file_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return keywords

# All search queries
def read_keywords_from_file(file_path):
    with open(file_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return keywords

pro_keywords = read_keywords_from_file("pro_ed_labels.txt")
con_keywords = read_keywords_from_file("con_ed_labels.txt")
no_of_samples_pro = maxResults * len(pro_keywords)
no_of_samples_con = maxResults * len(con_keywords)

# 1st wave of data: pro-ed
pro_titles = []
pro_descriptions = []
pro_ids = []
pro = []

for keyword in pro_keywords:
    req = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults = maxResults)
    res = req.execute()
    index = pro_keywords.index(keyword) + 1
    while(len(pro_titles)<(maxResults*index)): #ensures we get 50 by keyword
        for i in range(len(res['items'])):
            pro_titles.append(res['items'][i]['snippet']['title'])
            pro_descriptions.append(res['items'][i]['snippet']['description'])
            pro_ids.append(res['items'][i]['id']['videoId'])
            category.append(keyword)
            pro.append('pro')

        if('nextPageToken' in res):
            next_page_token = res['nextPageToken']
            req = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults = maxResults, pageToken=next_page_token)
            res = req.execute()
        else:
            break


# 2nd wave of data: con-ed
con_titles = []
con_descriptions = []
con_ids = []
con = []

#Alternate keys to ensure we don't go over quota
youtube_api = build('youtube','v3', developerKey = API_KEY2)

for keyword in con_keywords:
    req = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults=maxResults)
    res = req.execute()
    index = con_keywords.index(keyword) + 1
    while(len(con_titles)<(maxResults*index)):
        for i in range(len(res['items'])):
            con_titles.append(res['items'][i]['snippet']['title'])
            con_descriptions.append(res['items'][i]['snippet']['description'])
            con_ids.append(res['items'][i]['id']['videoId'])
            category.append(keyword)
            con.append('con')

        if('nextPageToken' in res):
            next_page_token = res['nextPageToken']
            req = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults = maxResults, pageToken=next_page_token)
            res = req.execute()
        else:
            break

# Construct Dataset
final_titles = pro_titles + con_titles
final_descriptions = pro_descriptions + con_descriptions
final_ids = pro_ids + con_ids
side = pro + con #just keeps count of which videos are pro and which are con
data = pd.DataFrame({'Video Id': final_ids, 'Title': final_titles, 'Description': final_descriptions, 'Category': category, 'Pro_or_Con': side})
data.to_csv('updated_collected_data_raw.csv')
