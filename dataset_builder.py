#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from googleapiclient.discovery import build
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd

#GLOBAL VARS
maxResults = 20 #want 20 videos per keyword
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



# In[13]:


def get_comments(id, api_key):
    
    # retrieve youtube video results
    comments = ""
    replies = ""
    youtube = build('youtube', 'v3',
                    developerKey=api_key)
    try:
        video_response=youtube.commentThreads().list(
            part='snippet,replies',
            videoId=id
        ).execute()

        while video_response:
            for item in video_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                # get num of reply of comment
                replycount = item['snippet']['totalReplyCount']
                # if reply is there, get it
                if replycount>0:
                    for reply in item['replies']['comments']:
                        reply = reply['snippet']['textDisplay']
                        replies += reply
                # add comment to big comment string, along with its replies
                comments += " " + comment +" "+ replies
                # empty reply string
                replies = ""

            # repeat for next page
            if 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part = 'snippet,replies',
                    videoId = id,
                    pageToken = video_response['nextPageToken']
                ).execute()
            else:
                break 
        return comments
    except: #returns error if comments have been disabled, empty string is like video with 0 comments
        return " "
   
    
# Allows to get complete description (more than 160 chars)
def get_full_desc(id, api_key):
    """
    Get complete description of a YouTube video.

    Parameters:
    - id (str): The video ID.
    - api_key (str): The YouTube Data API key.

    Returns:
    str: The complete description of the video.
    """

    scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(
        part="snippet",
        id=id
    )
    response = request.execute()

    # access description from snippet
    description = response['items'][0]['snippet']['description']

    return description

#Reads all keywords from label file
def read_keywords_from_file(file_path):
    """
    Read keywords from a file.

    Parameters:
    - file_path (str): The path to the file containing keywords.

    Returns:
    list: A list of keywords.
    """
    with open(file_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return keywords

def gather_data(api_key1, api_key2, api_key3, keywords, side_ed):
    """
    Gather data from YouTube API for a given set of keywords.

    Parameters:
    - api_key (str): YouTube Data API keys.
    - keywords (list): A list of keywords.
    - side_ed (int): 0 if con-ed, 1 in pro-ed contant scraped.

    Returns:
    tuple: Lists of video titles, descriptions, video IDs, and categories.
    """
    youtube_api = build('youtube', 'v3', developerKey=api_key1)
    titles = []
    descriptions = []
    comments = []
    ids = []
    side = []
    info = []
    markup = []

    for keyword in keywords:
        req = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults=maxResults)
        res = req.execute()
        index = keywords.index(keyword) + 1

        while len(titles) < (maxResults * index):
            for i in range(len(res['items'])):
                titles.append(res['items'][i]['snippet']['title'])
                descriptions.append(get_full_desc(res['items'][i]['id']['videoId'], api_key2))
                comments.append(get_comments(res['items'][i]['id']['videoId'], api_key3))
                ids.append(res['items'][i]['id']['videoId'])
                category.append(keyword)

                side.append(side_ed)
                info.append(1-side_ed) #assumed all con-ed were informative, all pro-ed non-informative
                markup.append(side_ed) # all pro-ed contained markup-words like tw and recovery help (misleading terms), all con-ed did not

            if 'nextPageToken' in res:
                next_page_token = res['nextPageToken']
                req = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults=maxResults,
                                                pageToken=next_page_token)
                res = req.execute()
            else:
                break

    return titles, descriptions, ids, side, info, markup, comments


# In[5]:


comment1 = get_comments("0s6ohWO6YXY",API_KEY1)
print(comment1[:200]) #prints all comments


# In[15]:


comment2 = get_comments("FEY9miq9wkY",API_KEY1)
print(comment2) #video where comments are disabled (kids content)


# In[17]:


comment3 = get_comments("8rZP8ef7fps",API_KEY1)
print(comment3) #video with no comments


# In[19]:


# read keywords, store in list
pro_keywords = read_keywords_from_file("pro_ed_labels.txt")
con_keywords = read_keywords_from_file("con_ed_labels.txt")

#step 1: pro-ed data with 1 api key
pro_titles, pro_descriptions, pro_ids, pro_side, pro_info, pro_markup, pro_comments = gather_data(API_KEY1, API_KEY2, API_KEY3, pro_keywords, 1)

#step 2: con-ed data with other api key
con_titles, con_descriptions, con_ids, con_side, con_info, con_markup, con_comments = gather_data(API_KEY4, API_KEY5, API_KEY6, con_keywords, 0)


# In[20]:


# Construct Dataset
final_titles = pro_titles + con_titles
final_descriptions = pro_descriptions + con_descriptions
final_comments = pro_comments + con_comments
final_ids = pro_ids + con_ids
side = pro_side + con_side
info = pro_info + con_info
markup = pro_markup + con_markup

print(len(final_titles), len(final_descriptions), len(final_comments), len(final_ids), len(side), len(info), len(markup))
data = pd.DataFrame({'Video Id': final_ids, 'Title': final_titles, 'Description': final_descriptions, 'Comments' : final_comments,
                     'Query': category, 'Pro_or_Con': side, 'Informative': info, 'Markup_terms': markup})
data.to_csv('raw_data_with_comments.csv')


# In[ ]:


# Now we want to try and perform sentiment analysis using comments

