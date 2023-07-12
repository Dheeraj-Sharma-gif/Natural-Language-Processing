#!/usr/bin/env python
# coding: utf-8

# # Implementing Parts of Speech

# In[1]:


get_ipython().system('pip install spacy')


# In[1]:


import spacy


# In[7]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[2]:


nlp = spacy.load('en_core_web_sm')


# In[3]:


doc = nlp(u"I will work on robotics")


# In[4]:


doc.text


# In[5]:


doc[0]


# In[6]:


doc[0].pos_


# In[7]:


doc[0].tag_


# In[8]:


doc[-1]


# In[9]:


spacy.explain('PRP')


# In[10]:


for word in doc:
    print(word.text,'------->',word.pos_,word.tag_,spacy.explain(word.tag_))


# In[11]:


from spacy import displacy
displacy.render(doc,style='dep',jupyter=True)


# In[12]:


options = {
    'distance': 80,
    'compact': True,
    'color': '#fff',
    'bg': 'green'
}


# In[13]:


displacy.render(doc,style='dep',jupyter=True,options=options)

