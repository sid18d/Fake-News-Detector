#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Loading necessary libraries


# In[2]:


FakeNews = pd.read_csv("news.csv")
print(FakeNews.head())

# Loading the data 


# In[3]:


x = np.array(FakeNews["title"])
y = np.array(FakeNews["label"])

# Labeling the Columns

cv = CountVectorizer()
x = cv.fit_transform(x)

# Preparing data to train the model


# In[4]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Separate the dataset into training and testing sets

model = MultinomialNB()

# Using  Multinomial Naive Bayes algorithm

model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# Training the Model


# In[5]:


news_headline = "CA Exams 2021: Supreme Court asks ICAI to extend opt-out option for July exams, final order tomorrow"
FakeNews = cv.transform([news_headline]).toarray()
print(model.predict(FakeNews))

# Testing the Model using random news title from google search


# In[8]:


news_headline = "Camel urine can cure Corona Virus"
FakeNews = cv.transform([news_headline]).toarray()
print(model.predict(FakeNews))

# Testing the Model using randon fake news headline


# In[ ]:




