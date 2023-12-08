#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries for dataclean
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import os
import pandas as pd


# In[3]:


# Importing tools for preprocessing and analysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np


# In[5]:


#Import Data
data = pd.read_csv('./raw_data_with_comments.csv', encoding = 'utf-8-sig')
data.head(7)


# In[7]:


####STEP 1: DATACLEAN####

data['Title'] = data['Title'].map(lambda x: re.sub(r'\d+', '', str(x)))
data['Description'] = data['Description'].map(lambda x: re.sub(r'\d+', '', str(x)))
data['Comments'] = data['Comments'].map(lambda x: re.sub(r'\d+', '', str(x)))

data['Title'] = data['Title'].map(lambda x: x.lower())
data['Description'] = data['Description'].map(lambda x: x.lower())
data['Comments'] = data['Comments'].map(lambda x: x.lower())
data['Comments'][4][1500:1800] #WTS @@username and emoji pb


# In[9]:


import emoji
#removes usernames and taggs of other channels
def remove_username_substring(input_string):
    # in scrapped comments, username look like '@ @ user_name'
    pattern = r'@\s*@\s*\w+'
    result_string = re.sub(pattern, '', input_string)
    return result_string
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

data['Comments'] = data['Comments'].apply(remove_username_substring)
data['Comments'] = data['Comments'].apply(remove_emoji)
data['Description'] = data['Description'].apply(remove_emoji)
data['Title'] = data['Title'].apply(remove_emoji)
data['Comments'][4][1500:1800] #@@username and emoji pb solved


# In[11]:


data['Title']  = data['Title'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
data['Description']  = data['Description'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
data['Comments']  = data['Comments'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))


data['Title'] = data['Title'].map(lambda x: x.strip())
data['Description'] = data['Description'].map(lambda x: x.strip())
data['Comments'] = data['Comments'].map(lambda x: x.strip())
data.head(7)

#from langdetect import detect
thought about potentially taking off non english comments but might also take off terms langdetect doesn't know
such as 'mia' 'ana' 'ed' which are relevant to our study
# In[13]:


data['Title'] = data['Title'].map(lambda x: word_tokenize(x))
data['Description'] = data['Description'].map(lambda x: word_tokenize(x))
data['Comments'] = data['Comments'].map(lambda x: word_tokenize(x))

#takes off links and tags people put in the description/comments
data['Description'] = data['Description'].map(lambda lst: [word for word in lst if not word.startswith("https")])
data['Comments'] = data['Comments'].map(lambda lst: [word for word in lst if not word.startswith("https")])
data['Description'] = data['Description'].map(lambda lst: [word for word in lst if not word.startswith("href")])
data['Comments'] = data['Comments'].map(lambda lst: [word for word in lst if not word.startswith("href")])
data['Description'] = data['Description'].map(lambda lst: [word for word in lst if not word.startswith("@")])

data['Title'] = data['Title'].map(lambda x: [word for word in x if word.isalpha()])
data['Description'] = data['Description'].map(lambda x: [word for word in x if word.isalpha()])
data['Comments'] = data['Comments'].map(lambda x: [word for word in x if word.isalpha()])

stop_words = set(stopwords.words('english'))
data['Title'] = data['Title'].map(lambda x: [w for w in x if not w in stop_words])
data['Description'] = data['Description'].map(lambda x: [w for w in x if not w in stop_words])
data['Comments'] = data['Comments'].map(lambda x: [w for w in x if not w in stop_words])

#word lemmatization for verbs and nouns
lem = WordNetLemmatizer()
data['Title'] = data['Title'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
data['Description'] = data['Description'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
data['Comments'] = data['Comments'].map(lambda x: [lem.lemmatize(word,"v") for word in x])

data['Title'] = data['Title'].map(lambda x: [lem.lemmatize(word,"n") for word in x])
data['Description'] = data['Description'].map(lambda x: [lem.lemmatize(word,"n") for word in x])
data['Comments'] = data['Comments'].map(lambda x: [lem.lemmatize(word,"n") for word in x])

data.head(7)


# In[17]:


#turn list back to string
data['Title'] = data['Title'].map(lambda x: ' '.join(x))
data['Description'] = data['Description'].map(lambda x: ' '.join(x))
data['Comments'] = data['Comments'].map(lambda x: ' '.join(x))
data.head(7)


# In[19]:


#for videos that do not have a description
data['Description'] = data['Description'].replace(['nan'], [''], regex=True)
data.head(7)


# In[23]:


#Add information on comment/feeback emotion
import text2emotion as te


# In[ ]:


data['Sentiment_Comments'] = data['Comments'].map(lambda x: max(te.get_emotion(x), key=te.get_emotion(x).get))
data.head(7)

Now we want to create learning functions to analyze sentiment, eating disorder keywords, 
non-ED keywords.
# In[114]:


import snorkel
from snorkel.labeling import labeling_function
from textblob import TextBlob
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
import matplotlib.pyplot as plt
from snorkel.labeling import LFAnalysis
from snorkel.preprocess import preprocessor
from snorkel.classification.data import DictDataset, DictDataLoader


# In[115]:


ED_keywords = ["binge", "binging", "purge", "purging", "fat", "anorexic", "ana", "mia", "bulimia", "trigger warning", "tw","weight", "eating", "compulsive", "lifestyle", "body check", "starving", "fat", "hate", "wish", "skinny", "thinspo"]
tw_keywords = ["tw", "tw ed", "twed", "gethelp", "seek", "help", "medical"]
pro_ED_keywords = ["fat", "restriction", "restrict", "lifestyle","lowcalorie", "thyn","meanspo","thinspo","thin","skinny","disgust","belly", "slender","debloat","highres","restrict"]
con_ED_keywords = ["unglamorizing", "fantasizing", "recovery","disorder","illness","neutrality", "intuitive","positivity"]                 


# In[116]:


ABSTAIN = -1
PRO = 1
CON = 0


# In[117]:


#take text from title + desc / and text from title + desc + comments
csv = pd.DataFrame()
csv['label'] = data.apply(lambda row: str(row['Pro_or_Con']), axis=1)
csv['text'] = data.apply(lambda row: str(row['Title']) + " " + str(row['Description']), axis=1)
csv['text_with_coms'] = data.apply(lambda row: str(row['Title']) + " " + str(row['Description']) + " " + str(row['Comments']), axis=1)
csv.head(2)


# In[118]:


text = csv.text.tolist()
text_with_coms = csv.text_with_coms.tolist()                              
labels = csv.label.tolist()
newlabels = []
newtext = []
newtextcoms = []
for i in range(len(text)):
    newtext.append(str(text[i]))
    newtextcoms.append(str(text_with_coms[i]))
    newlabels.append(int(labels[i]))


# In[119]:


df = pd.DataFrame({"labels": newlabels, "text": newtext});
df_train,df_test = train_test_split(df,train_size=0.9)
df_test = pd.concat([df_test, pd.read_csv("raw_data_with_comments.csv")]) #test data


# In[ ]:


@labeling_function()
def contains_proEDkeywords(x):
    string = str(x.text).lower()
    for keyword in pro_ED_keywords: 
        if keyword in string:
            return PRO
    return ABSTAIN

@labeling_function()
def contains_conEDkeywords(x):
    string = str(x.text).lower()
    for keyword in con_ED_keywords: 
        if keyword in string:
            return CON
    return ABSTAIN

@labeling_function()
def contains_tw_keywords(x):
    string = str(x.text).lower()
    for keyword in tw_keywords:
        if keyword in string:
            return CON
    return ABSTAIN

#allows us to still label videos with pro-ed themes but that circumvent using pro-recovery terms
@labeling_function()
def markup(x):
    string = str(x.text).lower()
    for keyword in tw_keywords:
        if x.labels == 1 and keyword in string:
            return PRO
        elif keyword in string:
            return CON
    return ABSTAIN

#we labled pro-ed as 1 if video was found using a proED query
@labeling_function()
def prelabel(x):
    if x.labels == 1:
        return PRO
    else:
        return CON

@labeling_function()
#returns negative if text sentiment polarity is greater than 0.3, and positive otherwise
def textblob_polarity(x):
    return CON if TextBlob(x.text.lower()).sentiment.polarity > 0.3 else PRO

@labeling_function()
def emotion(x):
    emotion_map = te.get_emotion(x.text)
    if (emotion_map['Sad'] + emotion_map['Fear'] + emotion_map['Angry'])  > 0.9:
        return PRO
    elif (emotion_map['Happy'] > .3):
        return CON
    elif (emotion_map['Surprise'] > .6):
        return CON
    elif emotion_map['Sad'] > .5:
        return PRO
    elif emotion_map['Angry'] > .5:
        return PRO
    elif emotion_map['Fear'] > .5:
        return PRO
    else:
        return ABSTAIN

#all labeling functions
lfs = [contains_proEDkeywords, markup, contains_conEDkeywords, contains_tw_keywords, emotion, prelabel, textblob_polarity]
lfs2 = [contains_proEDkeywords, contains_conEDkeywords, emotion, prelabel, textblob_polarity]


# In[137]:


####STEP 2.1: USE TF IDF FOR KEYWORDS####
le = LabelEncoder()
le.fit(data.Query)
data.Query = le.transform(data.Query)


# In[139]:


# TF-IDF => high score = keywords / important descriptors
tfidf_title = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
tfidf_desc = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
tfidf_coms = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
labels = data.Query


# In[141]:


features_title = tfidf_title.fit_transform(data.Title).toarray()
features_description = tfidf_desc.fit_transform(data.Description).toarray()
features_coms = tfidf_coms.fit_transform(data.Comments).toarray()
print('Title Features Shape: ' + str(features_title.shape))
print('Description Features Shape: ' + str(features_description.shape))
print('Comments Features Shape: ' + str(features_coms.shape))


# In[143]:


####STEP 3.1: UNIGRAM+BIGRAM ANALYSIS####
# Plotting class distribution
data['Query'].value_counts().sort_values(ascending=False).plot(kind='bar', y='Number of Samples',title='Number of samples for each class')
plt.show()


# In[185]:


def read_keywords_from_file(file_path):
    with open(file_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return keywords

def get_side(keyword):
    if keyword in read_keywords_from_file('pro_ed_labels.txt'):
        return 1
    else:
        0    


# In[207]:


# Get 10 best keywords for each keyword, Title features (we only print the first 2)
N = 10
count1 = 0

title_common_words_pro = []
title_common_words_con = []

MAX_PRINT = 3
for current_class in list(le.classes_):

    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_title, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_title.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    
    if get_side(current_class):
        #pro ed
        title_common_words_pro += unigrams[-N:]
        title_common_words_pro += bigrams[-N:]
    else:
        title_common_words_con += unigrams[-N:]
        title_common_words_con += bigrams[-N:]
    
    if count1 > MAX_PRINT:
        continue
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")
    count1 += 1

print("**********************************************************************************************************************************")



# In[213]:


title_common_words_pro[:10]


# In[219]:


# Get 10 best keywords for each keyword, Desc features (we only print the first 2)
N = 10
count2 = 1

desc_common_words_pro = []
desc_common_words_con = []

for current_class in list(le.classes_):
    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_description, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_desc.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    
    
    if get_side(current_class):
        #pro ed
        desc_common_words_pro += unigrams[-N:]
        desc_common_words_pro += bigrams[-N:]
    else:
        desc_common_words_con += unigrams[-N:]
        desc_common_words_con += bigrams[-N:]
    
    if count2 > MAX_PRINT:
        continue
        
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")
    count2 += 1


# In[225]:


desc_common_words_con[25:35]


# In[227]:


# Get 10 best keywords for each keyword, Comments features
N = 20
count1 = 1
count2 = 1

coms_common_words_pro = []
coms_common_words_con = []

MAX_PRINT = 2
for current_class in list(le.classes_):

    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_coms, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_coms.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    
    if get_side(current_class):
        #pro ed
        coms_common_words_pro += unigrams[-N:]
        coms_common_words_pro += bigrams[-N:]
    else:
        coms_common_words_con += unigrams[-N:]
        coms_common_words_con += bigrams[-N:]
    
    if count1 > MAX_PRINT:
        continue
    
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")
    count1 += 1
    
print("**********************************************************************************************************************************")


# In[ ]:


coms_common_words_pro[65:75]


# In[ ]:


# import libraries
import collections
import hashlib
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag, map_tag
from os import path
import pandas as pd
from scipy.misc import imread
import string
import random
import re
from wordcloud import WordCloud, STOPWORDS


# In[ ]:


# choose the most relevant words to consider,
# according to tags
coms_common_words_pro
coms_common_words_con
taggedListpro = []
taggedListcon = []

for item in coms_common_words_pro:
    item = pos_tag(item)
    taggedListpro.append(item)
    
for item in coms_common_words_con:
    item = pos_tag(item)
    taggedListcon.append(item)
   


# In[ ]:


relevantListpro = []
relevantListcon = []
count = 0
for i in taggedListpro:
    for j in i:
        if j[1] == "NN" or j[1] == "JJ" or j[1] == "JJR" or j[1] == "JJS":
            relevantListpro.append(j[0].lower()) 
for i in taggedListcon:
    for j in i:
        if j[1] == "NN" or j[1] == "JJ" or j[1] == "JJR" or j[1] == "JJS":
            relevantListcon.append(j[0].lower()) 


# In[ ]:


finalListPro = FreqDist(relevantListpro)
commonPro = finalListPro.most_common()
uncommonPro = list(reversed(finalListPro.most_common()))[:50]
print("These are the most common words for pro-ED videos:",commonPro, "\n")
print("These are the most uncommon words for pro-ED videos:", uncommonPro, "\n")


# In[ ]:


finalListCon = FreqDist(relevantListcon)
commonCon = finalListCon.most_common()
uncommonCon = list(reversed(finalListCon.most_common()))[:50]
print("These are the most common words for con-ED videos:",commonCon, "\n")
print("These are the most uncommon words for con-ED videos:", uncommonCon, "\n")


# In[ ]:


#Create wordcloud for pro
completeText = ""
for key, val in commonPro:
    completeText += (key + " ") * val
    
text = completeText
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      relative_scaling = 0.5,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


#Create wordcloud for pro
completeText = ""
for key, val in commonCon:
    completeText += (key + " ") * val
    
text = completeText
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      relative_scaling = 0.5,
                      stopwords = 'to of'
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


completeText = ""
for key, val in newCommon:
    completeText += (key + " ") * val


# In[ ]:


completeText = ""
for key, val in newCommon:
    completeText += (key + " ") * val


# In[103]:


#apply the labeling functions to each row of the training data df_train, produce a label matrix L_train
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)


# In[39]:


print(L_train)


# In[ ]:


#Started trying to use Hugginface's MentalBert as a classifier


# In[45]:


from snorkel.labeling import LFAnalysis

analysis_df = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

analysis_df.to_csv('./firstPassLearningFuncsAnalysis.csv', index = False, header=True)


# In[47]:


from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)


# In[49]:


from snorkel.labeling import filter_unlabeled_dataframe

probs_train = label_model.predict_proba(L=L_train)

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)
print(df_train_filtered)
df_train_filtered.to_csv('./firstPassSnorkelLabel.csv', index = False, header=True)


# In[60]:


#Now we try using mental bert 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertForSequenceClassification, BertTokenizer
import torch


# In[95]:


from huggingface_hub import login
login("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")
tokenizer = AutoModelForSequenceClassification.from_pretrained("mental/mental-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("mental/mental-roberta-base")


# In[96]:


from datasets import load_dataset, DatasetDict
dataset = load_dataset('csv', data_files='firstPassSnorkelLabel.csv')


# In[80]:


dataset = DatasetDict(
    train=dataset['train'].shuffle(seed=1111).select(range(1172)), 
    val=dataset['train'].shuffle(seed=1111).select(range(937, 1054)),
    test=dataset['train'].shuffle(seed=1111).select(range(1054, 1172)),
)#80-10-10
BATCH_SIZE = 16


# In[163]:


## LOSS PLOT ##
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.savefig('loss_plot_lstm.png')


# In[169]:


## ACCURACY PLOT ##

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')  
plt.plot(history.history['val_accuracy'], label='test')  
plt.legend()
plt.show()


# In[ ]:


"""
maybe go back and take off positive words from pro-ED things = markup terms
"""

