#!/usr/bin/env python
# coding: utf-8

# 

# # STEP 1 - Importing the data set

# In[5]:


#importing all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import warnings as wg
wg.filterwarnings("ignore")


# In[7]:


#Reading data from source link  

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)


# In[8]:


#observe the dataset

df.head()


# In[9]:




df.tail()


# In[ ]:


#to find number of column and rows

df.shape()

(25,2)


# In[12]:


#to find more information about given dataset

df.info()


# In[13]:


df.describe()


# In[14]:


#to check if given dataset is having missing or null values

df.isnull().sum()


# # STEP 2 - visualizing the data set

# In[17]:


#ploting dataset

plt.rcParams["figure.figsize"]=[16,9]
df.plot(x='Hours', y='Scores', style='.',color='red',markersize=15)
plt.title('Hours vs Percentage')
plt.xlabel('.....Hours Studied.....')
plt.ylabel('.....Percentage Scores....')
plt.grid()
plt.show()


# In[18]:


#to determine correlation between variables we can use .corr()

df.corr()


# # STEP 3- Data Preparation

# In[19]:


df.head()


# In[21]:


#using iloc function we can divide the dataset

x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values


# In[22]:


x


# In[23]:


#spliting data into training and testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)


# # STEP 4 - Training Algorithm

# In[30]:



from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


# # STEP 5- Visualizing the model

# In[33]:


line = model.coef_*x + model.intercept_

plt.rcParams["figure.figsize"] =[16,9]
plt.scatter(x_train, y_train, color='green')
plt.plot(x,line,color='purple');
plt.xlabel('....Hours Studied....')
plt.ylabel('....Percentage Score....')
plt.grid()
plt.show()


# In[34]:


plt.rcParams["figure.figsize"]=[16,9]
plt.scatter(x_test, y_test, color='green')
plt.plot(x,line,color='purple');
plt.xlabel('....Hours Studied....')
plt.ylabel('....Percentage Score....')
plt.grid()
plt.show()


# # STEP 6 - Making Prediction

# In[35]:


print(x_test)
y_pred = model.predict(x_test)


# In[36]:


#compare actual vs predicted
y_test


# In[37]:


y_pred


# In[39]:


#compare actual vs predicted

comp = pd.DataFrame({'Actual':[y_test],'predicted':[y_pred]})
comp


# In[40]:


#testing our own data

hours = 9.25
own_pred = model.predict([[hours]])
print("the predicted score if a person studied for",hours,"hours is",own_pred[0])


# # Evaluating tha model

# In[42]:


from sklearn import metrics
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




