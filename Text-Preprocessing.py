#!/usr/bin/env python
# coding: utf-8

# # Text-Processing 

# In[1]:


import nltk
nltk.download()


# In[2]:


import re
pattern = re.compile(r'<[^>]+>')
def remove_html(string):
    return pattern.sub('', string)
text="Ram is<taking @ very+ > STRONGLY"
new_text=remove_html(text)
print(f"Text without html tags: {new_text}")


# In[3]:


import re
pattern = re.compile(r'http://\S+|https://\S+')
def remove_url(string):
    return pattern.sub('', string)
text="This is a text with a URL https://www.java2blog.com/ to remove."
new_text=remove_url(text)
print(f"Text without html tags: {new_text}")


# In[4]:


import string
exclude=string.punctuation
def remove_punct(text):
      return text.translate(str.maketrans('', '', exclude))
    
sample='HE IS !,;;A OERJ'
remove_punct(sample)


# In[5]:


from textblob import TextBlob
incrct="certan umbrlla"
textblb=TextBlob(incrct)
textblb.correct()


# In[6]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(wordsFiltered)


# In[7]:


import re
def remove_emoji(text):
    emoji_pattern=re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji
remove_emoji("I loved thjis movie and Copy and 📋 Paste Emoji 👍 No apps req")


# In[8]:


import emoji
print(emoji.demojize('python is ✂️ Copy and 📋 Paste Emoji 👍 No apps req'))


# In[9]:


#tokenization
from nltk.tokenize import sent_tokenize, word_tokenize

data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
print(sent_tokenize(data))
print(word_tokenize(data))


# In[10]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem_words(text):
    return" ".join([ps.stem(word) for word in text.split()])

sample2="walk walked walks walking"
stem_words(sample2)


# In[11]:


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
sentence = "Python programmers often tend like programming in python because it's like english. We call people who program in python pythonistas."
punct = sentence.translate(str.maketrans("", "", string.punctuation))
word_tokens = word_tokenize(punct)
print("{0:20}{1:20}".format("--Word--","--Lemma--"))
for word in word_tokens:
   print("{0:20}{1:20}".format(word, wnl.lemmatize(word, pos="v")))


# In[12]:


import numpy as np
import pandas as pd
df=pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
bow=cv.fit_transform(df['text'])
print(cv.vocabulary_)
print(bow[0].toarray())
print(bow[1].toarray())
cv.transform(["Campusx watch and write comment of campusx"]).toarray()


# In[13]:


import numpy as np
import pandas as pd
df=pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(2,2))
bow=cv.fit_transform(df['text'])
print(cv.vocabulary_)
print(bow[0].toarray())
print(bow[1].toarray())
cv.transform(["Campusx watch and write comment of campusx"]).toarray()


# In[14]:


import numpy as np
import pandas as pd
df=pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(1,2))
bow=cv.fit_transform(df['text'])
print(cv.vocabulary_)
print(bow[0].toarray())
print(bow[1].toarray())
cv.transform(["Campusx watch and write comment of campusx"]).toarray()


# In[15]:


import numpy as np
import pandas as pd
df=pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(3,3))
bow=cv.fit_transform(df['text'])
print(cv.vocabulary_)
print(bow[0].toarray())
print(bow[1].toarray())
cv.transform(["Campusx watch and write comment of campusx"]).toarray()


# In[16]:


import numpy as np
import pandas as pd
df=pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
tfidf.fit_transform(df['text']).toarray()





# In[17]:


import numpy as np
import pandas as pd
df=pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
tfidf.fit_transform(df['text']).toarray()
print(tfidf.idf_)

