#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# # Data processing

# In[2]:


data = pd.read_csv('Copy of sonar data.csv', header = None)


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data[60].value_counts()


# In[7]:


data.groupby(60).sum()


# # Training & test data seperation

# In[ ]:


## seperating label from data


# In[8]:


X = data.drop(columns=60, axis=1)
Y = data[60]


# In[ ]:


# Training and testing data


# In[9]:


X_train,X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify = Y, random_state=1)


# # Creating logistic regression model

# In[ ]:


# Creating Logistic Regression model


# In[10]:


model = LogisticRegression()


# In[11]:


model.fit(X_train, Y_train)


# In[ ]:


# Evaluating the model


# In[ ]:


# Accuracy of training data
# Accuracy score ranging from 75 to above is considered to be good for the model


# In[12]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[13]:


print("training data accuracy:", training_data_accuracy)


# In[ ]:


# Accuracy of testing data


# In[15]:


X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[16]:


print("testing data accuracy:", testing_data_accuracy)


# In[23]:


input_data = (0.0091,0.0213,0.0206,0.0505,0.0657,0.0795,0.0970,0.0872,0.0743,0.0837,0.1579,0.0898,0.0309,0.1856,0.2969,0.2032,0.1264,0.1655,0.1661,0.2091,0.2310,0.4460,0.6634,0.6933,0.7663,0.8206,0.7049,0.7560,0.7466,0.6387,0.4846,0.3328,0.5356,0.8741,0.8573,0.6718,0.3446,0.3150,0.2702,0.2598,0.2742,0.3594,0.4382,0.2460,0.0758,0.0187,0.0797,0.0748,0.0367,0.0155,0.0300,0.0112,0.0112,0.0102,0.0026,0.0097,0.0098,0.0043,0.0071,0.0108)

# Changing tuple input to numpy array, and reshaping it because model will accept only one input at a time
nparray = np.asarray(input_data).reshape(1,-1)
prediction = model.predict(nparray)

if(prediction[0] == 'R'):
 print("Object is Rock")
elif(prediction[0] == "M"):
 print("Object id Mine !!!!!")    

