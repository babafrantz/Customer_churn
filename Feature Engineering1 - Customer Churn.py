#!/usr/bin/env python
# coding: utf-8

# ##### FEATURE ENGINEERNG

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[2]:


train_data= pd.read_csv(r"C:\Users\yomol\train_data.csv")
historical_data=pd.read_csv(r"C:\Users\yomol\train_data.csv")


# In[3]:


historical_data.head()


# In[4]:


train_data.head()


# ##### Handling Categorical Features

# In[5]:


# we can see that has gas is a  categorical feild and will require converting to numerical value for use in modelling.
# We will use the one-hot encoding method to achieve this
 
train_data = pd.get_dummies(train_data, columns=["has_gas"])


# In[6]:


train_data.head()


# #### FEATURE ENGINEERING

# Using the HeatMap to determine correlations between the features

# In[7]:


# We use heatmap to understand the correlations between data and to know the features to select. 

plt.figure(figsize = (15,10))
sns.heatmap(train_data.corr())


# from the above heatmap, we can see that features like consumptions(cons_12m, cons_gas_12m, cons_last_month), forecast_price_energy_p1, forecast_price_pow_p1, imp_cons, nb_prod_act, num_years_antig, has_gas_t are moving towards the negative correlation with churn. 

# ###### Selecting features and transforming features 

# We select features that have negative correlations for classification problems. This features based on domain knowledge also will influence churn. 

# In[8]:



features = ['cons_12m', 'cons_gas_12m', 'cons_last_month','num_years_antig','has_gas_t','imp_cons','nb_prod_act']
x = train_data.loc[:, features].values


# In[9]:


y = train_data.loc[:,['churn']].values


# In[10]:


x = StandardScaler().fit_transform(x)


# In[11]:


pd.DataFrame(data = x, columns = features).head()


# #### Principal component analysis

# In[12]:


# We do the pca based on the number of features

pca = PCA(n_components=7)


# In[13]:


principalComponents = pca.fit_transform(x)


# In[14]:


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5','principal component 6','principal component 7'])


# In[15]:


principalDf.head()


# In[16]:


train_data[['churn']].head()


# In[17]:


final_train_data = pd.concat([principalDf, train_data[['churn']]], axis = 1)
final_train_data


# ###### Visualising the principal components

# In[18]:


def principal_component_plot(pc1,pc2):
    
    
    fig = plt.figure(figsize = (8,8))
    ax  = fig.add_subplot(1,1,1) 
   
    ax.set_xlabel(pc1, fontsize = 15)

    ax.set_ylabel(pc2, fontsize = 15)


    ax.set_title('2 Component PCA', fontsize = 20)
    
    targets = [0,1]

    colors = ['r','g']
    
 

    for target, color in zip(targets,colors):
        indicesToKeep = final_train_data['churn'] == target
        first_column = final_train_data.loc[indicesToKeep, pc1]
        second_column = final_train_data.loc[indicesToKeep, pc2]
        ax.scatter(first_column,second_column, c = color, s = 50)
               
        ax.legend(targets)
        ax.grid()
    


# In[19]:



principal_component_plot('principal component 1','principal component 2')


# In[20]:


principal_component_plot('principal component 3','principal component 4')


# In[21]:


principal_component_plot('principal component 5','principal component 6')


# In[22]:


principal_component_plot('principal component 1','principal component 7')


# In[23]:



pca.explained_variance_ratio_


# ##### Plotting a Scree plot

# Scree plot tells us how many components to take in a PCA. 

# In[25]:


# Scree plot tells us how many components to take in a PCA. 

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# We will be picking three components of the PCA components. 

# ##### Final Data to be used in buildng the model.

# In[26]:


final_train_data = final_train_data[['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5', 'churn']]


# In[27]:


final_train_data


# In[29]:


final_train_data.to_csv("final_train_data.csv")


# In[ ]:




