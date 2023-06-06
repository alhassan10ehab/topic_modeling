#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries

import pandas as pd                                        
import numpy as np  
get_ipython().system('pip install missingno')
import missingno as msno
import matplotlib.pyplot as plt                           
import seaborn as sns  
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install bertopic')
import bertopic
from bertopic import BERTopic
from tqdm import tqdm

# the following code ensures that you can see your (print) results for multiple tasks within a coding block
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


pd.options.display.width = None


# Lab 1 contains 4 analysis questions. 
# The dataset is reused from tutorial part 2, i.e., the sampled_arxiv_cs.csv file.

# Question 1: Load in the provided dataset and summarize the basic statistics. Specifically your code should answer:
# 
# 1) What are the feature types and their basic statistics (using describe(include='all'))
# 2) How many features have missing values? 

# In[3]:


# read the data
df = pd.read_csv('sampled_arxiv_cs.csv')

# print the first five records 
df.head()

#check the features data type
df.info()

# check some statistical information about the data
df.describe(include="all")

# print data dimension 
df.shape


# In[4]:


# it seems that the data has null records in the last
df.tail()


# In[5]:


# print number of features that contain missing values
(df.isna().any()).sum()


# In[6]:


# Plot bar chart for columns to illustrate the null values 
msno.bar(df)

# plot null values as a matrix
msno.matrix(df)

# plot heat map for the columns to observe the relation between null values
msno.heatmap(df)

# print dendogram for the missing value
msno.dendrogram(df)


# In[7]:


# display number of nulls in each column
df.isna().sum()


# In[8]:


# print shape before droping missing value 
df.shape

# drop missing values based on two columns 
df.dropna(subset=['title', 'abstract'], inplace=True)

# print shape after droping missing value
df.shape


# In[9]:


# put your code for Q1 here, you can have multiple code blocks.


# Summarize your answers to Q1 below, based on your analysis:
# 
# 1- as illusterated in df.info() all the features type are object and when performing describe( include='all') it was observed as a statistical view that the most frequent category is 'cs.cv' with 3105 times 
# 
# 2- according to the above graphs and statistics it was observed that the number of features that have missing values are 14 and the missing values are dense at the tail of the data 

# Question 2: Preprocess the abstracts of the papers in the provided data corpus (note, in the tutorial, we use titles, not abstracts)
# Apply bertopic on the abstracts and explain the topics you received and discuss the quality of the topics and determine what is the optimal topic number for abstracts.

# In[10]:


# put your code for Q2 here, you can have multiple code blocks.

get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz')
    
get_ipython().system('pip3 install spacy')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 

from IPython.utils import io 
import en_core_sci_lg  # import downlaoded model
import string
get_ipython().system('pip install minisom')
from minisom import MiniSom  
from sklearn.cluster import SpectralClustering 
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS 


# In[11]:


# Now let's create a tokenizer 
parser = en_core_sci_lg.load()
parser.max_length = 7000000 #Limit the size of the parser
punctuations = string.punctuation #list of punctuation to remove from text
stopwords = list(STOP_WORDS)


def spacy_tokenizer(sentence):
    ''' Function to preprocess text of scientific papers 
        (e.g Removing Stopword and puntuations)'''
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ] # transform to lowercase and then split the scentence
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ] #remove stopsword an punctuation
    mytokens = " ".join([i for i in mytokens]) 
    return mytokens


# In[12]:


# Apply the tokenizer to process title, we will use the processed_text as the input for clustering task, but feel free to compare with just using the raw titles.
tqdm.pandas()

# Function to handle float values in the "title" column
def handle_float_value(value):
    if isinstance(value, float):
        return str(value)  # Convert float to string
    return value

# Apply the tokenizer to the "abstract" column
df["processed_text"] = df["abstract"].apply(handle_float_value).progress_apply(spacy_tokenizer)


# In[13]:


# now let's create a topic model using BERTopic 

# since we have quite a lot data points, let's start with a large topic number (50) and see how coherent are the learned topics.
topic_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L6-v2", min_topic_size=50)
topics, _ = topic_model.fit_transform(df["processed_text"].to_numpy()); len(topic_model.get_topic_info())


# In[14]:


# print information for this first 10 topics 
topic_model.get_topic_info().head(10)


# In[15]:


# represent the TF-IDF for words in a bar chart
topic_model.visualize_barchart(top_n_topics=10, height=700)


# In[16]:


# print the clusters for the first 10 topics 
topic_model.visualize_topics(top_n_topics=10)


# In[17]:


# print the clusters for the first 50 topics 
topic_model.visualize_topics(top_n_topics=50)


# In[18]:


# draw the hierarchical clustering
topic_model.visualize_hierarchy(top_n_topics=10, width=800)


# In[19]:


# draw the similarity matrix between the topics 
topic_model.visualize_heatmap(n_clusters=5, top_n_topics=50)


# Summarize your answers to Q2 below, based on your analysis:
# 
# i received 101 for number of topics which is not good and it is smaller than the true number of topics because we have 8908 unique value at the categorical feature so we had to get about 8908 topics 
# 
# at the first had a trial before the above trial with min_topic_size = 200 and the result was worst than when it was 50 because the number of topics was 30
# 
# (-1) topic indicates that the model did not identify these words
# 
# and it was observed that when i increase the number of min_topic_size the number of topics decrease 

# Question 3: Compare the topics learned by BERTopic with the original categories of the papers. 
# For instance, you can pick one popular category, such as cs.CV, and check how those papers belong to this category are cateogrized by your topic model learned in previous step. 

# In[20]:


# put your code for Q3 here, you can have multiple code blocks.

# convert the topics list into dataframe and print its shape
topicss= pd.DataFrame(topics)
topicss.shape


# In[21]:


# print the number of categories (real topics)
(df['categories'].nunique())


# In[22]:


# concatenate the two dataframes to add the topics column 
concatenated_df = pd.concat([df, topicss], axis=1)

# print the dimension of the data frame after concatenating 
concatenated_df.shape

# print the first five rows after concatenating
concatenated_df.head()


# In[23]:


# Identify the column to rename
column_to_rename = 0

# Use rename function to rename the topics column after concatenating
concatenated_df = concatenated_df.rename(columns={column_to_rename: 'topicss'})
concatenated_df.head()


# In[24]:


# select the records which their categorical value is 'cs.CV'
filtered_df = concatenated_df[concatenated_df['categories'] == 'cs.CV']

# print the dimension 
filtered_df.shape

# print the first five rows 
filtered_df.head()


# In[25]:


# print the number of topics for 'cs.CV' category 
filtered_df['topicss'].nunique()


# Summarize your answers to Q3 below, based on your analysis:
# 
# after analysis it is observed that we get a 50 topics for the 'cs.CV' category which is not good the predicted topics should be about one 

# Question 4: Redo the topic modeling using the same method, but only on the papers tagged by the category you chcked in Q3. For instance, in Q3, you pick cs.CV, then in Q4, you first filter data by category = cs.CV and then apply the topic modeling on the selected documents.
# Compare your topics found and your results from Q3. 

# In[27]:


# put your code for Q4 here, you can have multiple code blocks.

# perform the bertopic on the 'cs.CV' category only
topic_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L6-v2", min_topic_size=50)
topics, _ = topic_model.fit_transform(filtered_df["processed_text"].to_numpy()); len(topic_model.get_topic_info())


# In[28]:


# print the information about each topic
topic_model.get_topic_info().head(3)


# In[29]:


# represent the TF-IDF for each topic
topic_model.visualize_barchart(top_n_topics=3, height=700)


# In[30]:


# draw a cluster for each topic 
topic_model.visualize_topics(top_n_topics=1)


# In[31]:


# draw the hierarchal clustering for each topic 
topic_model.visualize_hierarchy(top_n_topics=3, width=800)


# In[32]:


# draw the similarity matrix between the topics 
topic_model.visualize_heatmap(n_clusters=1, top_n_topics=50)


# Summarize your answers to Q4 below, based on your analysis:
# 
# after analysis and performing the bertopic ont the 'cs.CV' it was divided into 2 topics only which is realy good because we have only one topic
# we can assume tha the first topic (0) talks about general computer vision and the second topic (1) talks about face recogination or image segmentation according to the representation of each topic 
# (-1) indicates that the model could not add this topic in any topic did not identify this words 

# In[ ]:




