#!/usr/bin/env python
# coding: utf-8

# In[ ]:



## https://raw.githubusercontent.com/AniSkywalker/SarcasmDetection/master/resource/train/Train_v1.txt


# In[15]:


import numpy as np   
import pandas as pd


# In[16]:


data = pd.read_csv("sarcasm_tweets.csv")
data.head()


# In[17]:


data.isnull().sum()


# In[18]:


data.shape


# In[19]:


X = data['Tweet'].values
y = data['Label'].values


# In[20]:


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[21]:


nltk.download("stopwords")


# In[22]:


ps = PorterStemmer()
corpus = []
for i in range(len(X)):
  tweets = re.sub('[^a-zA-Z]', ' ', X[i])
  tweets = tweets.lower()
  tweets = tweets.split()

  tweets = [ps.stem(word) for word in tweets if word not in stopwords.words('english')]
  tweets = ' '.join(tweets)
  corpus.append(tweets)


# In[23]:


X[0],corpus[0]


# In[39]:


from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(np.asarray(corpus), y, test_size = 0.2, random_state = 24)


# In[41]:


X_train.shape, X_test.shape


# In[42]:


X_train[0], y_train[0]


# In[43]:


X_test[0], y_test[0]


# In[44]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# In[45]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


# In[46]:


tfidf = TfidfVectorizer()
model = RandomForestClassifier()


# In[47]:


x_train = tfidf.fit_transform(X_train)
x_test = tfidf.transform(X_test)


# In[48]:


x_train


# In[49]:


model = RandomForestClassifier(bootstrap=False, class_weight='balanced',
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=250,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)


# In[50]:


pipe_line = Pipeline([
                      ('tfidf', tfidf),
                      ('model', model)
])


# In[60]:


pipe_line.fit(X_train, y_train)


# In[61]:


pred = pipe_line.predict(X_test)
print(accuracy_score(y_test, pred))
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# In[62]:


import pickle


# In[63]:


pickle.dump(pipe_line, open("randomforest.pkl", 'wb'))


# In[66]:


get_ipython().system('pip install xgboost==1.1.1')


# In[65]:


import xgboost
from xgboost import XGBClassifier


# In[67]:


xgb_model = XGBClassifier()
xgb_pipe_line = Pipeline([
                      ('tfidf', tfidf),
                      ('model', xgb_model)
])


# In[68]:


xgb_pipe_line.fit(X_train, y_train)


# In[69]:


pred = xgb_pipe_line.predict(X_test)
print(accuracy_score(y_test, pred))
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# In[70]:


pickle.dump(xgb_pipe_line, open("xgb.pkl", 'wb'))


# In[71]:


xgboost.__version__


# In[ ]:




