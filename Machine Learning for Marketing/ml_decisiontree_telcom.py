
# coding: utf-8

# ## Preparation for modeling

# In[76]:


import pandas as pd
import numpy as np


# In[42]:


telco_raw = pd.read_csv('telco.csv',';')


# In[43]:


telco_raw.head()


# In[44]:


telco_raw.dtypes


# ### Target and Features

# Separate the identifier and target variable names as lists

# In[45]:


custid = ['customerID']
target = ['Churn']


# Separate categorical and numeric column names as lists

# In[46]:


categorical = telco_raw.nunique()[telco_raw.nunique()<10].keys().tolist()
categorical.remove(target[0])

numerical = [col for col in telco_raw.columns
                if col not in custid+target+categorical]


# One-hot encoding categorical variables

# In[47]:


telco_raw = pd.get_dummies(data=telco_raw, columns=categorical, drop_first=True)


# In[48]:


telco_raw[numerical].dtypes


# ### Scaling

# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


scaler = StandardScaler()


# In[50]:


scaled_numerical = scaler.fit_transform(telco_raw[numerical])
scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical)


# In[51]:


telco_raw = telco_raw.drop(columns=numerical, axis=1)
telco = telco_raw.merge(right= scaled_numerical, how = 'left', left_index=True, right_index=True)


# In[52]:


telco.head()


# ## Modeling 

# In[58]:


features = [col for col in telco.columns
                if col not in custid+target]


# In[61]:


X = telco[features]
Y = telco[target]


# ### Supervised Learning

# In[62]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[66]:


train_X, test_X, train_Y, test_Y =train_test_split(X,Y, test_size=0.25)

print(train_X.shape[0] / X.shape[0])
print(test_X.shape[0] / X.shape[0])


# In[73]:


clf = tree.DecisionTreeClassifier(max_depth = 7, 
               criterion = 'gini', 
               splitter  = 'best')
treemodel = clf.fit(train_X, train_Y)


# In[74]:


pred_Y =  treemodel.predict(test_X)


# In[78]:


print("Training accuracy: ", np.round(clf.score(train_X, train_Y), 3)) 
print("Test accuracy: ", np.round(accuracy_score(test_Y, pred_Y), 3))


# ### Unpervised Learning

# In[55]:


from sklearn.cluster import KMeans


# In[80]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(telco[features])


# In[82]:


telco[features].assign(Cluster=kmeans.labels_)


# In[ ]:


#telco[features].groupby('Cluster').mean()

