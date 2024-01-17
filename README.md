# YouTube Recommendation Eating Disorder Content Analysis

Welcome to the repo containing my work for my individual research project for the COGS 401 course which investigates the impact of recommendation systems on information dissemination regarding eating disorders, with a specific emphasis on YouTube. 

# Table of Contents
1. [ Abstract ](#intro)
2. [ Repository Organization ](#repoorg)
2. [ Data Collection ](#datacoll)
3. [ Findings ](#find)


<a name="intro"></a>
## 1. Abstract

The research integrates various methodologies, including web scraping, analysis of recommendation algorithms, and survey experiments, to examine how YouTube may inadvertently foster environments conducive to pro-eating disorder (pro-ED) and anti-recovery content dissemination. A significant focus is placed on understanding the behavioral patterns and perceptions of users aged 19 to 32, a demographic particularly vulnerable to disordered eating and body image concerns.

The study reveals that user engagement with YouTube, especially the frequency of use and interaction with the platform’s recommendations, is closely linked to their perceptions and behaviors related to eating disorders and body image. Machine learning models, including Naive Bayes and LSTM, were employed to classify content into pro-ED and anti-eating disorder (con-ED) categories. The results indicate a nuanced complexity in text classification for such social issues, with simpler models outperforming more complex ones, underscoring the need for finer model tuning and larger datasets.

The study also proposes the implementation of advanced text classification systems using the Snorkel framework to enhance context-aware content moderation. This research contributes to the understanding of social media’s impact on mental health and underscores the importance of informed policy-making in the digital landscape.


<a name="repoorg"></a>
## 2. Repository Organization

Here is a brief overview of the key elements of this repo:

Data Collection Scripts (dataclean-comments.py, dataclean.ipynb, dataclean.py, dataset_builder.ipynb, dataset_builder.py): These files are essential for generating the dataset used in the study. The scripts and notebooks detail the process of collecting, cleaning, and preprocessing the data, laying the foundation for the subsequent analysis.

Model Development and Testing (current_unfinished_model_training.ipynb, model_training.pdf): The Jupyter notebook current_unfinished_model_training.ipynb contains the code for building and testing various binary classification models that categorize content into pro-ED and con-ED, model_training.pdf is a compiled summary of the results and findings from this notebook.

Survey Experiment Analysis (YouTube, Health Content, and Body Image Perception Study.pdf): This PDF document includes the analysis of the survey experiment.

Initial Model Exploration (firstPassLearningFuncsAnalysis.csv, firstPassSnorkelLabel.csv): These files represent preliminary explorations into fine-tuning the classification model. Although not fully developed, they contain valuable ideas that could potentially enhance the model's performance.

Label Files (con/pro_ed_labels.txt): contain the labels used for the classification task in the study, distinguishing between pro-ED and con-ED content.


<a name="datacoll"></a>
## 3. Data Collection

### We employed a multifaceted approach using various data collection and analysis tools. Here’s an overview:
#### Data collection
YouTube Data API: We utilized the YouTube Data v3 API, with eight different API keys from new Google accounts, to gather a broad and unbiased dataset of both pro-eating disorder (pro-ED) and anti-eating disorder (con-ED) content.
Manual Examination: Initial video examination to understand common themes and subthemes in pro-ED and con-ED content.
Snowball Sampling Method: Identification of specific harmful terms like "thinspiration", "skinny", and "size0" to refine our search.

#### Data pre-processing
Natural Language Toolkit (NLTK): Utilized for text processing, including tokenization, stopwords removal, and lemmatization.
Pandas: For data manipulation and analysis.
Regular Expressions & String Libraries: Employed for cleaning and preparing textual data.
TF-IDF Vectorization: Transformation of text data into a format suitable for machine learning models, using separate vectorizers for 'Title' and 'Description' columns.
Label Encoding: Encoding class labels with LabelEncoder.

#### ML Models
Classifier Models: We tested models like Naive Bayes, Support Vector Machine (SVM), AdaBoost, and Long Short-Term Memory (LSTM) networks, each offering unique strengths in pattern recognition.
Sci-kit Learn: Utilized for implementing traditional machine learning models and feature selection.
PyTorch & Transformers: Used for implementing and experimenting with LSTM and BERT models for sequence classification.

#### Data analysis tools
Seaborn & Matplotlib: For data visualization.
Statistical Analysis: Conducted using scipy and statsmodels for ANOVA and MANOVA.
Snorkel Framework: Proposed for advanced text classification and context-aware content moderation.
WordCloud: To visualize frequent terms in our dataset.

<a name="find"></a>
## 4. Findings
For more details about our findings please refer to the final report and data analysis.
