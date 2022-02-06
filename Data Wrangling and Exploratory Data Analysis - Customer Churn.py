#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling and Exploratory Data Analysis

# # Data Wrangling

# In[1]:


## Importing neccesary packages into the notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ##### Importing data from CSV file directory

# In[2]:


## Importing the CSV from file directory into the notebook

## Importing the customer data

Training_data = pd.read_csv(r"C:\Users\yomol\OneDrive\Desktop\BCG\ml_case_training_data (1).csv")

## Importing the pricing data
historical_data = pd.read_csv(r"C:\Users\yomol\OneDrive\Desktop\BCG\ml_case_training_hist_data (1).csv")

## Importing the churn data
churn_data = pd.read_csv(r"C:\Users\yomol\OneDrive\Desktop\BCG\ml_case_training_output (1).csv")


# ###### Understanding imported data

# In[3]:


# Viewing the imported Training_data and its row length and column length
Training_data


# In[4]:


# showing the whole historical_data dataframe and its row length and column length 

historical_data


# In[5]:


# Showing the whole of the churn_data dataframe showing the row length and column length. 
churn_data


# #### Combining Two Dataframes

# We can combine the churn data and training data on the "id" since they have the same row length.. 

# In[6]:


train_data = Training_data.merge(churn_data, on="id", how= "left")
train_data


# In[7]:


# Understanding the data types, the number of columns, number of rows, column label of the newly merged train_data etc
train_data.info()


# In[8]:


# Understanding the data types, the number of columns, number of rows, column label of the newly merged train_data etc
historical_data.info()


# we need to check statistics of each dataframe and also check if there are missing values
# 

# In[9]:


# Checking the descriptive statistics of the train_data

train_data.describe()


# ##### Observations from Descriptive statistics:
# 
# 1. The minimum for majority of the columns are negative number. It is possible that there is a debt of power. This will be handled later by replacing missing values with the median. 
# 
# 2. The 	"campaign_disc_ele" looks like an empty column. 

# In[10]:


historical_data.describe()


# ##### Observations from Descriptive statistics:
# 1. There are no empty columns
# 2. The minimum prices for some price heading or column are of negative value. We must find a way to make it possitive later when we deal with missing values. 

# ##### Observing the percentage of missing values in both dataframe train_data and historical_data. 

# In[11]:


# Writing a function to check for missing values

def percentage_of_missing_values(data):
    num_of_mis_val = data.isnull().sum()
    percent_miss= (num_of_mis_val/16096)*100
    return percent_miss

    


# In[12]:


percentage_of_missing_values(train_data)


# We can see that for "campaign_disc_ele" column, missing values are 100% meaning the entire column is missing. "date_first_activ, forecast_base_bill_ele,forecast_base_bill_year, forecast_bill_12m,forecast_cons" have 78% missing values. We can let them go since they are more than 75%. We will be doing that when its time to deal with missing values.         
#              

# In[13]:


percentage_of_missing_values(historical_data)


# The percentage of missing values in the "historical_data" column is more very insignificant and the missing values can be filled with the mean or median. 

# #### Data Visualisations

# ###### Churn Data

# We will start visualisations with the churn data. First we rename the id column with company_id for visualisation purpose. 

# In[14]:


train_data.head()


# In[15]:


# Renaming the id column. 

train_data.rename(columns={"id":"company_id"})


# In[16]:


train_data["churn"].value_counts()


# In[17]:


churn_percentage = (train_data["churn"].value_counts()/len(train_data.index))*100
churn_percentage


# In[18]:


# Creating a dataframe from the results of churn percentage. 
churn_data_frame = pd.DataFrame({"churn":[0,1],
                                 "Percentage":[90.09,9.9]})              


# In[19]:


sns.barplot(x="churn", y="Percentage", data=churn_data_frame, hue="churn")


# we can see from the above visualisation that only 9.9% of the customers have churn. While a 90.9 percent has not churn.

# ###### SME ACTIVITY VISUALISATIONS

# In[20]:


SME_activity=train_data[["id","churn","activity_new"]]

SME_activity


# In[21]:


SME_activity=SME_activity.groupby([SME_activity["activity_new"],
 SME_activity["churn"]])["id"].count().unstack(level=1).sort_values(by=[0], ascending=False)


# In[22]:


SME_activity.plot(kind="bar",figsize=(18, 10),width=3,stacked=True,  title="SME Activity")
# Labels
plt.ylabel("Number of companies")
plt.xlabel("Activity")
# Rename legend
plt.legend(["Retention", "Churn"], loc="upper right")
# Remove the label for the xticks as the categories are encoded and we can't draw any meaning from them yet
plt.xticks([])
plt.show()


# We can see from the above that few customers have churn due based on category of company activity. 

# ###### SALES CHANNEL VISUALISATIONS

# In[23]:


sales_channel = train_data[["id","channel_sales","churn"]]
sales_channel = sales_channel.groupby([sales_channel["channel_sales"],
sales_channel["churn"]])["id"].count().unstack(level=1).fillna(0)


# In[24]:


sales_channel_churn = (sales_channel.div(sales_channel.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)


# ###### CONSUMPTION VISUALISATIONS

# In[25]:


# We select all the columns that relates to consumption and id from the train data

consumption = train_data[["id","cons_12m", "cons_gas_12m","cons_last_month", "imp_cons", "has_gas", "churn"]]


# In[26]:


def cons_dist_plot(dataframe, column,bins_=50):
    # Create a dataframe grouping the consumption columns using the churning rate.
    churn_sep_column_df = pd.DataFrame({"Retention":dataframe[dataframe["churn"]==0][column],
                                       "churn":dataframe[dataframe["churn"]==1][column]})
    
    # plotting the histogram using the created dataframe
    churn_sep_column_df[["Retention","churn"]].plot(kind="hist",bins=bins_,stacked=True)

   
 
    
# Plotting the different histograms. 
cons_dist_plot(consumption,"cons_12m")
cons_dist_plot(consumption,"cons_gas_12m")
cons_dist_plot(consumption,"cons_last_month")
cons_dist_plot(consumption,"imp_cons")


# In[ ]:





# ##### DATA CLEANING

# In[27]:


train_data.head()


# In[28]:


# Plotting percentage of missing data
mis_percent = (train_data.isnull().sum()/len(train_data.index))*100

mis_percent.plot(kind="bar",  figsize=(18,10))
plt.xlabel("Variables")
plt.ylabel("Missing values (%)")
plt.show()


# In[29]:


# we drop columns with values more than 60%. 

train_data=train_data.drop(["activity_new","campaign_disc_ele","channel_sales","date_first_activ", "forecast_base_bill_ele", 
                 "forecast_base_bill_year","forecast_bill_12m","forecast_cons"],axis=1)


# In[30]:


# Columns where missing values are much has been dropped. 

train_data.info()


# In[31]:


# checking for duplicates

train_data[train_data.duplicated()]


# No column is duplicated in the train_data dataframe

# ##### REPLACING DATES

# In[32]:


# According to the model work, replacing missing values of dates will not work for dates and strings

train_data.loc[train_data["date_modif_prod"].isnull(),"date_modif_prod"] = train_data["date_modif_prod"].value_counts().index[0]
train_data.loc[train_data["date_end"].isnull(),"date_end"] = train_data["date_end"].value_counts().index[0]
train_data.loc[train_data["date_renewal"].isnull(),"date_renewal"] = train_data["date_renewal"].value_counts().index[0]


# ###### MISSING DATA FOR HISTORICAL DATA

# In[33]:


mis_percent_historical= (historical_data.isnull().sum()/len(historical_data.index))*100

mis_percent_historical.plot(kind="bar",figsize=(18,10))
plt.xlabel("Variables")
plt.ylabel("Missing values (%)")
plt.show()
                            


# we move to replace the 0.7 percent of the missing values of the historical_data with the median

# In[34]:


historical_data.loc[historical_data["price_p1_var"].isnull(),"price_p1_var"] = historical_data["price_p1_var"].median()
historical_data.loc[historical_data["price_p2_var"].isnull(),"price_p2_var"] = historical_data["price_p2_var"].median()
historical_data.loc[historical_data["price_p3_var"].isnull(),"price_p3_var"] = historical_data["price_p3_var"].median()
historical_data.loc[historical_data["price_p1_fix"].isnull(),"price_p1_fix"] = historical_data["price_p1_fix"].median()
historical_data.loc[historical_data["price_p2_fix"].isnull(),"price_p2_fix"] = historical_data["price_p2_fix"].median()
historical_data.loc[historical_data["price_p3_fix"].isnull(),"price_p3_fix"] = historical_data["price_p3_fix"].median()


# ###### FORMATTING DATES

# For effective use of the date in the prediction model, it is important to convert the date to datetime type.

# In[35]:


# Transform date columns to datetime type
train_data["date_activ"] = pd.to_datetime(train_data["date_activ"], format='%Y-%m-%d')
train_data["date_end"] = pd.to_datetime(train_data["date_end"], format='%Y-%m-%d')
train_data["date_modif_prod"] = pd.to_datetime(train_data["date_modif_prod"], format='%Y-%m-%d')
train_data["date_renewal"] = pd.to_datetime(train_data["date_renewal"], format='%Y-%m-%d')


# We need to do same for the historical data. 

# In[36]:


historical_data["price_date"] = pd.to_datetime(historical_data["price_date"], format='%Y-%m-%d')


# In[37]:


fig, axs = plt.subplots(nrows=7, figsize=(18,50))
# Plot boxplots
sns.boxplot((train_data["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train_data[train_data["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train_data["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train_data["forecast_cons_12m"].dropna()), ax=axs[3])
#sns.boxplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.boxplot((train_data["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train_data["imp_cons"].dropna()), ax=axs[6])
plt.show()


# ##### NEGATIVE VALUES

# In[38]:


historical_data.describe()


# we can see minimum values are negative. We will replace the minimum values with the median. 

# In[39]:


historical_data[(historical_data.price_p1_fix < 0) | (historical_data.price_p2_fix < 0) | (historical_data.price_p3_fix < 0)]


# In[40]:


historical_data.loc[historical_data["price_p1_fix"] < 0,"price_p1_fix"] = historical_data["price_p1_fix"].median()
historical_data.loc[historical_data["price_p2_fix"] < 0,"price_p2_fix"] = historical_data["price_p2_fix"].median()
historical_data.loc[historical_data["price_p3_fix"] < 0,"price_p3_fix"] = historical_data["price_p3_fix"].median()


# In[41]:


train_data.info()


# In[45]:


train_data.to_csv("train_data.csv")
historical_data.to_csv("historical_data.csv")


# In[ ]:




