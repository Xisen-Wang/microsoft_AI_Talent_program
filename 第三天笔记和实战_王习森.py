#!/usr/bin/env python
# coding: utf-8

# In[7]:


import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = "28198fb347e3451aabca72831fec42bb", "chinaeast2"
language = 'zh-CN'
def tts_speaker(text):
    speech_config=speechsdk.SpeechConfig(subscription=speech_key,region=service_region)#subscription就知道了是谁
      #构造语音合成服务
    speech_synthesizer=speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = speech_synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("识别结果 [{}]".format(text))
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("识别取消: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("错误: {}".format(cancellation_details.error_details))
        
tts_speaker("Hello World")


# In[6]:


tts_speaker("Hello World")


# In[25]:


#descent loss
import numpy as np
import matplotlib.pyplot as plt
import math


a = 1
b = 1
c = 1
x = np.array([1,2,3,4,5,6,7,8,9])
y = a*x.dot(x)+b*x+c



fig,ax=plt.subplots()
ax.plot(x, y, label=r"Track $m_1$")
plt.ylabel(r"$y$")
plt.xlabel(r"$x$")
ax.legend()
plt.show()



print(y)
gradient_a=0
gradient_b=0

deltaA = -x
deltaB = -1


# In[26]:


机器学习
线性回归用于分类
缺点：1.中间区域不够敏感，容易收到数据干扰

Logistic Regression用于分类
1.很平滑
2.0.5的地方很敏感
3.sigmoidal永远的神

逻辑回归模型用于多分类：One vs All分类器
1.A vs Others
2.B vs Others
3.C vs Others


# In[27]:


import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras


import matplotlib.style #Some style nonsense
import matplotlib as mpl #Some more style nonsense


#Set default figure size
#mpl.rcParams['figure.figsize'] = [12.0, 8.0] #Inches... of course it is inches
mpl.rcParams["legend.frameon"] = False
mpl.rcParams['figure.dpi']=200 # dots per inch


# # 实战！

# In[71]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[72]:


import warnings

warnings.filterwarnings('ignore')


# In[73]:


data = '/Users/wangxiang/Desktop/sklearn-ml-lab/heart.csv'

df = pd.read_csv(data, header=None)


# In[74]:


# view dimensions of dataset

df.shape


# In[75]:


# preview the dataset

df.head(50)


# In[76]:


#review its info
df.info()


# In[77]:


col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach','exang','oldpeak','slope','ca','thal2','target']

df.columns = col_names

col_names


# In[82]:


df.head()

df.drop(0,axis=0)


# for i in range(len(df['age'])): 
#     if int(df['age'][i]) > 50:
#         df['age']=1
#     else:
#         df ['age']=0
# df['age'].head()

# In[83]:


for col in col_names:
    
    print(df[col].value_counts())   


# In[84]:


df['target'].value_counts()


# In[85]:


# check missing values in variables

df.isnull().sum()


# In[93]:


X = df.drop(['target'], axis=1)

y = df['target']


# In[94]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)


# In[95]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[106]:


# check data types in X_train

X_train.dtypes


# In[110]:


# import category encoders

import category_encoders as ce

# encode variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg','thalach','exang','oldpeak','slope','ca','thal2'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[111]:


# import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier


# In[112]:


# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)


# In[113]:


y_pred_gini = clf_gini.predict(X_test)


# In[114]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))


# In[115]:


y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini


# In[116]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))


# In[117]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))


# In[118]:


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train)) 


# In[119]:





# In[ ]:




