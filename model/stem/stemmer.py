
# coding: utf-8

# In[24]:

import pandas as pd


# In[25]:

import nltk


# In[26]:

path = '/home/anurag/Desktop/random/test.tsv'


# In[27]:

df = pd.read_table(path, sep = '\t', header = None)


# In[28]:

#df.head()


# In[29]:

list = []
ps = nltk.stem.PorterStemmer()


# In[30]:

for word in df.loc[:, 0]:
    if type(word) is str:
        word = ps.stem(word)
    list.append(word)


# In[31]:

#list


# In[32]:

se = pd.Series(list)


# In[33]:

se
df.insert(loc = 0, column = '0', value = se)


# In[34]:

#df.head()


# In[35]:

#df.head()


# In[36]:

df = df.drop(df.columns[1], axis = 1)


# In[37]:

#df.head()


# In[38]:

df.to_csv('stem_test.tsv', sep = '\t', header = None)


# In[ ]:



