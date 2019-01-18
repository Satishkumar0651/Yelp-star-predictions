
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os 
os.chdir("E:/1ST SEM/eng/edwisor_assignments/19.nlp_yelp")


# In[2]:

df=pd.read_csv("yelp.csv")


# In[3]:

df.head(2)


# In[4]:

df.describe()


# In[5]:

df['text length'] = df['text'].apply(len)


# In[6]:

df['text length'].head()


# In[7]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:

g = sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text length')


# In[9]:

g=sns.FacetGrid(df,col='stars')
g.map(plt.hist,'cool')


# In[10]:

sns.boxplot(x='stars',y='text length',data=df,palette='rainbow')


# In[11]:

sns.countplot(x='stars',data=df,palette='rainbow')


# In[12]:

stars = df.groupby('stars').mean()
stars


# In[13]:

stars.corr()


# In[14]:

sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# In[15]:

Custexp = []
for i in df['stars']:
    if (i == 1):
        Custexp.append('BAD')
    elif (i == 3) | (i == 2):
        Custexp.append('NEUTRAL')
    else:
        Custexp.append('GOOD')
        

df['Customer_experience'] = Custexp
df['Customer_experience'].value_counts()
df['Text length'] = df['text'].apply(lambda x:len(x.split()))


# In[16]:

x = df['text']
y = df['Customer_experience']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# In[17]:

import nltk
nltk.download("stopwords")


# In[18]:

import string
import nltk
from wordcloud.wordcloud import WordCloud, STOPWORDS
from PIL import Image


# In[19]:

from nltk.corpus import stopwords
import string
def text_clean(message):
    nopunc = [i for i in message if i not in string.punctuation]
    nn = "".join(nopunc)
    nn = nn.lower().split()
    nostop = [words for words in nn if words not in stopwords.words('english')]
    return(nostop)


# In[20]:

good = df[df['Customer_experience'] == 'GOOD']
bad = df[df['Customer_experience'] == 'BAD']
neu = df[df['Customer_experience'] == 'NEUTRAL']


# In[21]:

good_bow = text_clean(good['text'])


# In[22]:

bad_bow = text_clean(bad['text'])


# In[23]:

neu_bow = text_clean(neu['text'])


# In[24]:

good_para = ' '.join(good_bow)
bad_para = ' '.join(bad_bow)
new_para = ' '.join(neu_bow)


# In[25]:

from wordcloud.wordcloud import WordCloud, STOPWORDS
from PIL import Image


# In[26]:

stopwords = set(STOPWORDS)
stopwords.add('one')
stopwords.add('also')
mask_image = np.array(Image.open("images.png"))
wordcloud_good = WordCloud(colormap = "Paired",mask = mask_image, font_path =None, width = 30, height = 20, scale=2,max_words=1000, stopwords=stopwords)
wordcloud_good.generate(good_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_good, interpolation="bilinear", cmap = plt.cm.autumn)
plt.axis('off')
plt.figure(figsize = (10,6))
plt.show()
wordcloud_good.to_file("good.png")


# In[27]:

stopwords = set(STOPWORDS)
wordcloud_neu = WordCloud(colormap = "plasma",font_path = None, width = 1100, height = 700, scale=2,max_words=1000, stopwords=stopwords).generate(new_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_neu,cmap = plt.cm.autumn)
plt.axis('off')
plt.show()
wordcloud_neu.to_file('neu.png')


# In[28]:

from sklearn.feature_extraction.text import CountVectorizer
cv_transformer = CountVectorizer(analyzer = text_clean)


# In[31]:

X = df['text']


# In[33]:

X = cv_transformer.fit_transform(X)


# In[36]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# In[37]:

from sklearn.naive_bayes import MultinomialNB


# In[38]:

nb = MultinomialNB()
nb.fit(x_train, y_train)


# In[ ]:

##predictions = nb.predict(x_test)
##predictions


# In[ ]:

##from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:

##print(confusion_matrix(y_test, predictions))
##print("\n")
##print(classification_report(y_test, predictions))


# In[ ]:

##from sklearn.ensemble import RandomForestClassifier


# In[ ]:

##rf = RandomForestClassifier(criterion='gini')
##rf.fit(x_train, y_train)


# In[ ]:

##pred_rf = rf.predict(x_test)


# In[ ]:

##print("Confusion Matrix\n",confusion_matrix(y_test, pred_rf))
##print("\n")
##print("Classification report\n",classification_report(y_test, pred_rf))

