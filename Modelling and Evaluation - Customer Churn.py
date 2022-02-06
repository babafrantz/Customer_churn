#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[2]:


final_train_data = pd.read_csv(r"C:\Users\yomol\final_train_data.csv")


# In[3]:


final_train_data


# In[4]:


del final_train_data["Unnamed: 0"]


# In[5]:


final_train_data


# ##### LOGISTIC REGRESSION MODELLING

# Arranging the data to independent and target variables. 

# In[9]:


X = final_train_data[["principal component 1","principal component 2","principal component 3","principal component 4","principal component 5"]]
y = final_train_data["churn"]


# Splitting the data train and test data

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# Prediction

# In[13]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# ##### Model Evaluation

# We need a confussion matrix to evalute the model.

# In[14]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# Classification report

# In[15]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# ROC Curve

# In[16]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

