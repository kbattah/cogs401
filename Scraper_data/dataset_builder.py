from googleapiclient.discovery import build
import pandas as pd

def search_videos_by_keyword(api, query, max_results=50, next_page_token=None):
    titles = []
    descriptions = []
    ids = []

    req = api.search().list(q=query, part='snippet', type='video', maxResults=max_results, pageToken=next_page_token)
    res = req.execute()

    for i in range(len(res['items'])):
        titles.append(res['items'][i]['snippet']['title'])
        descriptions.append(res['items'][i]['snippet']['description'])
        ids.append(res['items'][i]['id']['videoId'])

    next_page_token = res.get('nextPageToken')

    return titles, descriptions, ids, next_page_token

def gather_data_for_keyword(api, keyword, category, max_samples=1700):
    titles = []
    descriptions = []
    ids = []

    next_page_token = None

    while len(titles) < max_samples and next_page_token is not None:
        keyword_titles, keyword_descriptions, keyword_ids, next_page_token = search_videos_by_keyword(api, keyword, next_page_token=next_page_token)

        titles.extend(keyword_titles)
        descriptions.extend(keyword_descriptions)
        ids.extend(keyword_ids)

        category.extend([category] * len(keyword_titles))

    return titles, descriptions, ids

def gather_data_for_keywords(api, keywords):
    final_titles = []
    final_descriptions = []
    final_ids = []
    category = []

    for keyword in keywords:
        category_name = keyword  # can be changed, name of cat = keyword, might look weird
        keyword_titles, keyword_descriptions, keyword_ids = gather_data_for_keyword(api, keyword, category_name)
        final_titles.extend(keyword_titles)
        final_descriptions.extend(keyword_descriptions)
        final_ids.extend(keyword_ids)

    return final_titles, final_descriptions, final_ids, category

# Define the 2 lists of keywords from txt file computed previously
def read_keywords_from_file(file_path):
    with open(file_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return keywords

pro_keywords = read_keywords_from_file("pro_ed_labels.txt")
con_keywords = read_keywords_from_file("con_ed_labels.txt")

# Set up the YouTube API & call the function to gather data
api_key = "AIzaSyAh1sVjvm7b7GH0_W-1AXYOFldVvgRM9Gk"
#api_key = "AIzaSyAfIe8O946wN41Bla1JnRAGooASlNWMX1I"
youtube_api = build('youtube', 'v3', developerKey=api_key)

final_titles, final_descriptions, final_ids, category = gather_data_for_keywords(youtube_api, pro_keywords)

# Construct the dataset
data = pd.DataFrame({'Video Id': final_ids, 'Title': final_titles, 'Description': final_descriptions, 'Category': category})
data.to_csv('Collected_data_raw.csv')
