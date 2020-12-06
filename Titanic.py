#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas_profiling import ProfileReport


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


dataset=pd.read_csv('titanic.csv')


# In[4]:


len(dataset)


# In[5]:


profile=ProfileReport(dataset,title="Titanic Report")


# In[6]:


profile.to_file("Titanic Report.html")


# In[ ]:





# As per report there are 107 (12.0%) duplicate rows

# In[7]:


dataset.drop_duplicates(subset=None, keep='first',inplace=True)


# In[8]:


dataset.head(3)


# In[9]:


dataset.drop(['class','who','embark_town','embarked'],axis=1,inplace=True)


# In[10]:


dataset.drop(['deck','sibsp','parch'],axis=1,inplace=True)


# In[11]:


dataset['fare'].replace(to_replace =0,value =np.nan,inplace=True) 


# In[12]:


dataset.describe()


# In[13]:


dataset['fare'].fillna(value=dataset['fare'].median(),inplace=True)


# In[14]:


dataset.fillna(value=dataset['age'].mean(),inplace=True)


# In[15]:


dataset.isna().sum()


# In[16]:


dataset.head(3)


# In[17]:


dataset['adult_male']=dataset['adult_male'].astype(int)
dataset['alone']=dataset['alone'].astype(int)


# In[18]:


dataset=pd.get_dummies(dataset,drop_first=True)


# In[19]:


dataset.head(3)


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


plt.figure(figsize=(14,8))
plt.plot(dataset);
plt.legend(dataset)


# In[22]:


dataset['survived'].value_counts()


# In[23]:


print(f'class 0(Didnt survive:{100*(461/float(dataset.shape[0]))} % \n Class 1(Survived):{100*(323/float(dataset.shape[0]))} %')


# In[24]:


dataset.shape


# In[25]:


dataset.head()


# In[26]:


X=dataset.iloc[:,1:]
y=dataset['survived']


# In[27]:


dataset['survived'].value_counts()


# In[28]:


from imblearn.over_sampling import SMOTE


# In[29]:


oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


# In[30]:


y.value_counts()


# In[31]:


X=X.values
y=y.values


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[33]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# In[34]:


y_pred = classifier.predict(X_test)


# In[35]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr=classification_report(y_test,y_pred)
print(cm)
print("Accuracy Score=",accuracy_score(y_test,y_pred))
print("\n",cr)


# In[36]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr=classification_report(y_test,y_pred)
print(cm)
print("Accuracy Score=",accuracy_score(y_test,y_pred))
print("\n",cr)


# In[37]:


from sklearn.model_selection import GridSearchCV 


# In[38]:


# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  


# In[39]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


# In[40]:


grid.fit(X_train, y_train)


# In[41]:


print(grid.best_params_)


# In[42]:


print(grid.best_estimator_)


# In[43]:


grid_predictions = grid.predict(X_test)


# In[44]:


print(classification_report(y_test, grid_predictions)) 


# In[45]:


print(confusion_matrix(y_test, grid_predictions)) 


# In[46]:


import joblib


# In[47]:


joblib.dump(grid, 'titanic.pkl')


# In[ ]:




