#!/usr/bin/env python
# coding: utf-8

# # Text Classification for Harm and Non-Harm Twitter: A Comprehensive Guide to NLP with Machine Learning  Including Web Model Deployment

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# In[4]:


from wordcloud import WordCloud


# In[5]:


data=pd.read_csv("MeTooHate.csv")


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data.columns


# In[9]:


data.duplicated().sum()


# In[10]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(40,30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# In[11]:


def summary(data):
    print(f"Dataset has {data.shape[1]} features and {data.shape[0]} examples.")
    summary = pd.DataFrame(index=data.columns)
    summary["Unique"] = data.nunique().values
    summary["Missing"] = data.isnull().sum().values
    summary["Duplicated"] = data.duplicated().sum()
    summary["Types"] = data.dtypes
    return summary


# In[12]:


summary(data)


# In[13]:


def percent_value_counts(data, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(data.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(data.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# In[14]:


percent_value_counts(data, 'category')


# In[15]:


# Drop the unnecesary column from dataset
data = data.drop(['location'], axis=1)


# In[16]:


# Assuming 'data' is your DataFrame with text data
data = data.dropna(subset=['text'])


# In[17]:


summary(data)


# In[18]:


data.head()


# In[19]:


import matplotlib.cm


# In[20]:


data['category'].unique()


# In[21]:


data['category'].value_counts()


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

labels = ['0: Non-Harm', '1: Harm']
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='category', palette=['blue', 'red'])
plt.title('Hate or non-hate Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=labels)
plt.show()




# In[ ]:





# In[23]:


'''
Two find the Percentage of the label data and visulize with pie chart.
in the pie chart 88% of the data is not horm and and reaming 11% is harmfull tweets.
'''
data['category'].value_counts()\
.sort_values(ascending=False)\
.plot(kind='pie',explode=[0.2,0.05],
    labels=['non-hatred','hatred'],
    colors=['yellow','gray'],
    autopct='%1.2f%%',
    shadow=True,title="Percentage of the harm and not-harm twittes",figsize=(16,5))
plt.show()


# In[24]:


#Visualize the followers with status_id and 
data.groupby(['status_id'])['followers_count'].sum()[:50]\
.sort_values(ascending=False)\
.plot(kind='bar',figsize=(17,5),title="To visualize the follwer of the each member",color='r')
plt.xlabel("status_id")
plt.ylabel("Count of values")
plt.show()


# In[25]:


#Visualize the followers with status_id and 
data.groupby(['status_id'])['followers_count'].sum()[:20]\
.sort_values(ascending=False)\
.plot(kind='bar',figsize=(17,5),title="To visualize the follwer of the each member",color='g')
plt.xlabel("status_id")
plt.ylabel("Count of values")
plt.show()


# In[26]:


data_count = data['followers_count'].value_counts().sort_values(ascending=False)[:50]

plt.figure(figsize=(10, 6))
sns.barplot(x=data_count.index, y=data_count, color='green')

plt.title('To visualize followers_count')
plt.xlabel('followers_count')
plt.ylabel('Count of followers')
plt.show()


# In[ ]:





# In[27]:


data['text'] = data['text'].astype(str)
data['text_length'] = data['text'].apply(len)
data_length_count = data['text_length'].value_counts()[:50]

plt.figure(figsize=(12, 6))
sns.barplot(x=data_length_count.index, y=data_length_count, palette='Reds')

plt.title("Visualize the chartest is length in the data")
plt.xlabel("Length of the text")
plt.ylabel("Count of the values")
plt.show()



# In[ ]:





# In[28]:


#Visualize the text length in the text data
data['text']=data['text'].astype(str)
data['text_length']=data['text'].apply(len)
data['text_length'].value_counts()[:50].plot(kind='bar',figsize=(18,8),title="Visualize the chartest is length in the data",color='r')
plt.xlabel("Length of the text")
plt.ylabel("Count of the values")
plt.show()


# # Feature Engineering

# In[29]:


data['created_at'] = pd.to_datetime(data['created_at'])


# In[30]:


data['Day'] = data['created_at'].dt.day
data['Month'] = data['created_at'].dt.month
data['Year'] = data['created_at'].dt.year


# In[31]:


data


# In[32]:


summary(data)


# In[33]:


# Group by day and calculate average sentiment score
daily_statuses_count= data.groupby(data['Day'])['statuses_count'].mean()

# Visualize daily sentiment scores
plt.figure(figsize=(10, 6))
plt.plot(daily_statuses_count.index, daily_statuses_count.values)
plt.xlabel('Date')
plt.ylabel('Average Statuses Count')
plt.title('Daily Statuses ')
plt.show()


# In[34]:


# Group by day and calculate average sentiment score
daily_favorite_count= data.groupby(data['Day'])['favorite_count'].mean()

# Visualize daily sentiment scores
plt.figure(figsize=(10, 6))
plt.plot(daily_favorite_count.index, daily_favorite_count.values)
plt.xlabel('Date')
plt.ylabel('Average Favorite Count')
plt.title('Daily Favorite ')
plt.show()


# In[35]:


data['Year']


# In[36]:


# Group by day and calculate average sentiment score
Year_statuses_count= data.groupby(data['Year'])['statuses_count'].mean()

# Visualize daily sentiment scores
plt.figure(figsize=(10, 6))
plt.plot(Year_statuses_count.index, Year_statuses_count.values)
plt.xlabel('Year')
plt.ylabel('Average Statuses Count')
plt.title('Year Statuses ')
plt.show()


# In[37]:


summary(data)


# In[38]:


data['Month'].unique()


# In[39]:


'''Coverted the date time column into date and month and year format
   and extract the year and month and visualize the which year most twitted
'''

data['Year'].value_counts().plot(kind='pie',explode=[0.2,0.05],
    labels=['2018','2019'],
    colors=['red','gray'],
    autopct='%1.2f%%',
    shadow=True,title="Percentage of the twitter in the year wise",figsize=(16,5))
plt.show()


# In[40]:


import plotly.express as px


# In[41]:


import plotly.express as px

data['Year'] = data['Year'].astype(str)
data['Year'] = data['Year'].apply(lambda x: '2018' if x == '2018' else '2019')

fig = px.histogram(data, x='Year', color='Year', title='Number of Twitter in the Year',
                   labels={'Year': 'Year'},
                   category_orders={'Year': ['2018', '2019']})
fig.update_layout(showlegend=False)
fig.show()


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
sns.barplot(x=data['Year'], y=data['statuses_count'], data=data, ci=None, palette='hls')
plt.title('Number of Statuses in the Year')
plt.xlabel('Year')
plt.ylabel('Number of Statuses')
plt.show()


# In[43]:


data


# In[44]:


follwer_max=data['followers_count'].max()
follwer_min=data['followers_count'].min()
friend_max=data['friends_count'].max()
friend_min=data['friends_count'].min()
print(f'the highest followers in the data is {follwer_max} ')
print(f'the highest followers in the data is {follwer_min}')
print(f'the highest friends in the data is {friend_max}')
print(f'the highest friends in the data is {friend_min}')


# In[45]:


data1=data.copy()


# In[46]:


data1


# In[47]:


data1.columns


# In[48]:


data1=data[['text','category', 'Day', 'Month', 'Year']]


# In[49]:


data1 = data[['text', 'category', 'Day', 'Month', 'Year']]
data1['category_label'] = data1['category'].map({0: 'Not Harm', 1: 'Harm'})


# In[50]:


data1


# In[51]:


summary(data1)


# # Data Preprocessing / Data Cleaning

# In[52]:


def clean_Text(Text):
    Text = Text.lower() 
    return Text.strip()


# In[53]:


data1.text = data1.text.apply(lambda x: clean_Text(x))


# In[54]:


data1['text']


# In[55]:


import string
string.punctuation


# In[56]:


def remove_punctuation(Text):
    punctuationfree="".join([i for i in Text if i not in string.punctuation])
    return punctuationfree


# In[57]:


data1.text= data1['text'].apply(lambda x:remove_punctuation(x))


# In[58]:


data1.text


# In[59]:


import re
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens


# In[60]:


data['text']= data1['text'].apply(lambda x: tokenization(x))


# In[61]:


data['text']


# In[62]:


import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


# In[63]:


def remove_stopwords(Text):
    output= " ".join(i for i in Text if i not in stopwords)
    return output


# In[64]:


data1['text']= data1['text'].apply(lambda x:remove_stopwords(x))


# In[65]:


data1['text']


# In[66]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# In[67]:


nltk.download('wordnet')


# In[68]:


nltk.download('omw-1.4')


# In[69]:


def lemmatizer(Text):
    lemm_Text = "".join([wordnet_lemmatizer.lemmatize(word) for word in Text])
    return lemm_Text


# In[70]:


data1['text']=data1['text'].apply(lambda x:lemmatizer(x))


# In[71]:


data1['text']


# In[72]:


def clean_text(Text):
    Text = re.sub('\[.*\]','', Text).strip() 
    Text = re.sub('\S*\d\S*\s*','', Text).strip()  
    return Text.strip()


# In[73]:


data1['text'] = data1.text.apply(lambda x: clean_text(x))


# In[74]:


data1['text']


# In[75]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[76]:


stopwords = nlp.Defaults.stop_words
def lemmatizer(Text):
    doc = nlp(Text)
    sent = [token.lemma_ for token in doc if not token.Text in set(stopwords)]
    return ' '.join(sent)


# In[77]:


stopwords = nlp.Defaults.stop_words
def lemmatizer(text):
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)


# In[78]:


stopwords = nlp.Defaults.stop_words

def lemmatizer(Text):
    doc = nlp(Text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)


# In[79]:


data1['text'] =  data1.text.apply(lambda x: lemmatizer(x))


# In[80]:


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


# In[81]:


data1['text'] = data1.text.apply(lambda x: remove_urls(x))


# In[82]:


data1['text']


# In[83]:


def remove_digits(Text):
    clean_Text = re.sub(r"\b[0-9]+\b\s*", "", Text)
    return(Text)


# In[84]:


def remove_digits(Text):
    clean_Text = re.sub(r"\b[0-9]+\b\s*", "", Text)
    return clean_Text


# In[85]:


data1['text'] = data1.text.apply(lambda x: remove_digits(x))


# In[86]:


data1['text']


# In[87]:


def remove_digits1(sample_Text):
    clean_Text = " ".join([w for w in sample_Text.split() if not w.isdigit()]) 
    return(clean_Text)


# In[88]:


data1['text'] = data1.text.apply(lambda x: remove_digits1(x))


# In[89]:


data1['text']


# In[90]:


def remove_emojis(data):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', data)


# In[91]:


import re

def remove_emojis(Text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', Text)


# In[92]:


data1['text'] = data['text'].astype(str).apply(lambda x: remove_emojis(x))


# In[93]:


data1['text']


# In[94]:


data1


# In[95]:


data1['text_Length'] = data1['text'].apply(lambda x: len(x))


# In[96]:


data1


# In[97]:


plt.figure(figsize=(15,6))
sns.histplot(data1['text_Length'], kde = True, bins = 20, palette = 'hls')
plt.xticks(rotation = 0)
plt.show()


# In[98]:


import wordcloud


# In[100]:


from wordcloud import WordCloud
data = data1['text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[101]:


data = data1[data1['category_label']=="Harm"]['text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[102]:


data = data1[data1['category_label']=="Not Harm"]['text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[103]:


data2 = data1[['text','category_label']]


# In[104]:


data2


# In[105]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[106]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier


# In[107]:


X = data2['text']
y = data2['category_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[108]:


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[109]:


naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vectorized, y_train)


# In[110]:


y_pred = naive_bayes.predict(X_test_vectorized)


# In[111]:


accuracy = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)


# In[112]:


print("Accuracy:", accuracy)
print("Classification Report:\n", cr)


# In[113]:


param_grid = {
    'alpha': [0.1, 0.5, 1.0],  # Smoothing parameter
    'fit_prior': [True, False]  # Whether to learn class prior probabilities or not
}


# In[114]:


from sklearn.model_selection import GridSearchCV


# In[115]:


grid_search = GridSearchCV(estimator=naive_bayes, param_grid=param_grid, cv=5)
grid_search.fit(X_train_vectorized, y_train)


# In[116]:


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[140]:


best_model


# In[117]:


y_pred = best_model.predict(X_test_vectorized)


# In[118]:


accuracy = accuracy_score(y_test, y_pred)


# In[119]:


accuracy


# In[120]:


cr = classification_report(y_test, y_pred)


# In[121]:


print(cr)


# In[122]:


data3 = data1[['text','category_label']]


# In[123]:


data3


# In[139]:


summary(data3)


# In[124]:


import xgboost as xgb


# In[125]:


from sklearn.preprocessing import LabelEncoder


# In[127]:


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data3['category_label'])


# In[128]:


X = data3['text']
y = y_encoded


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[130]:


vectorizer1 = TfidfVectorizer()
X_train_vectorized = vectorizer1.fit_transform(X_train)
X_test_vectorized = vectorizer1.transform(X_test)


# In[131]:


xg_model = xgb.XGBClassifier()
xg_model.fit(X_train_vectorized, y_train)


# In[132]:


y_pred = xg_model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report)


# In[ ]:


# Random Forest


# In[133]:


model = RandomForestClassifier()


# In[134]:


model = RandomForestClassifier()
model.fit(X_train_vectorized, y_train)


# In[135]:


y_pred = model.predict(X_test_vectorized)


# In[136]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# In[137]:


import pickle

with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


# In[138]:


with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


# In[ ]:





# In[ ]:


# xgboost model 


# In[143]:


import pickle

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(xg_model, file)


# In[144]:


with open('tfidf_vectorizer1.pkl', 'wb') as file:
    pickle.dump(vectorizer1, file)


# In[147]:


import pickle

with open('rf_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[148]:


with open('tfidf_vectorizer1.pkl', 'wb') as file:
    pickle.dump(vectorizer1, file)

