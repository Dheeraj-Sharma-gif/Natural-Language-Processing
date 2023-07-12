#!/usr/bin/env python
# coding: utf-8

# # Word2Vec On Game of Thrones Story

# In[7]:


import numpy as np
import pandas as pd


# In[8]:


get_ipython().system('pip install gensim')


# In[9]:


import gensim
import os


# In[12]:


from nltk.tokenize import sent_tokenize
df['games'] = pd.read_csv(r'p1.csv')
sentences = (df['games']).apply(sent_tokenize)
print(sentences)


# In[13]:


len(df['games'])


# In[14]:


model = gensim.models.Word2Vec(
        window = 10,
        min_count = 2
 )


# In[17]:


model.build_vocab(df['games'])


# In[18]:


model.train(df['games'], total_examples=model.corpus_count, epochs = model.epochs)


# In[19]:


model.wv.get_normed_vectors()


# In[20]:


model.wv.get_normed_vectors().shape


# In[21]:


y = model.wv.index_to_key


# In[22]:


len(y)


# In[23]:


from sklearn.decomposition import PCA


# In[24]:


pca = PCA(n_components=3)


# In[25]:


X = pca.fit_transform(model.wv.get_normed_vectors())


# In[26]:


X[:5]


# In[27]:


import plotly.express as px
fig = px.scatter_3d(X[:100],x=0,y=1,z=2, color=y[:100])
fig.show()

