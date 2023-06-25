#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("D:\Data+Science@Consoleflare\Pandas\Python Customer segmentation and Clustering\Mall_Customers.csv")


# In[3]:


df.head()


# In[4]:


df.rename(columns={'Genre': 'Gender'}, inplace=True)


# # Univariate Analysis

# Univariate analysis is a statistical method used to examine and describe data involving a single variable. It focuses on understanding the characteristics, patterns, and distributions of a single variable in isolation, without considering the relationships with other variables. It provides valuable insights into the individual variables' behavior, including measures of central tendency, dispersion, and shape.
# 
# Customer segmentation and clustering involve dividing a customer base into distinct groups based on specific characteristics or behaviors. Univariate analysis can be utilized as a preliminary step to understand and identify relevant variables for customer segmentation and clustering. Let's consider an example to illustrate this process:
# 
# Suppose a retail company wants to segment its customer base to improve marketing strategies. They have collected data on various customer attributes, such as age, income, and spending habits. To begin the analysis, the company may conduct univariate analysis on each variable individually.
# 
# Age: The company examines the age distribution of its customers. They calculate measures such as the mean, median, and mode to understand the central tendency of age. Additionally, they analyze the range, standard deviation, and variance to assess the dispersion or variability of customer ages. This analysis helps identify the age groups that are most prevalent among customers.
# 
# Income: Univariate analysis is performed on income data to understand the income distribution among customers.
# 
# Spending habits: The company analyzes variables related to customer spending, such as average transaction amount or purchase frequency. They calculate measures like mean, median, and quartiles to understand the central tendency of spending habits.

# In[5]:


df.describe()


# In[6]:


df.columns


# The distplot(), kdeplot(), and catplot() functions are all part of the seaborn library and serve different purposes for data visualization.
# 
# distplot()function was used in previous versions of seaborn to plot the distribution of a univariate dataset.
# It combined a histogram with a kernel density estimate (KDE) plot.
# 
# The kdeplot() function is used to plot the kernel density estimate (KDE) of a univariate or bivariate dataset.
# It displays the shape of the distribution by estimating the underlying probability density function (PDF).
# It can be used to visualize the smoothness and pattern of data without explicitly binning it into a histogram.
# kdeplot() can also show rug plots, shading, and cumulative distribution functions (CDFs) as additional visual elements.
# catplot():
# 
# The catplot() function is used for categorical data visualization, particularly for plotting categorical variables against one or more numerical variables.
# It can be used to create various types of categorical plots such as box plots, violin plots, swarm plots, etc.
# catplot() provides a high-level interface for creating grouped or faceted plots based on the kind parameter, which allows you to specify the type of categorical plot you want to create.

# In[7]:


columns = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[8]:


columns = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(df[i],shade=True,hue=df['Gender'])


# In[9]:


columns = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    sns.catplot(x='Gender',y=i,data=df,kind='box')
    plt.show()


# In[10]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[11]:


sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df,hue='Gender')


# In[12]:


sns.pairplot(data=df)


# ### Dropping CustomerID

# In[13]:


# Check if 'CustomerID' column exists in the DataFrame
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# Create pairplot
sns.pairplot(data=df, hue='Gender')


# In[16]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()


# In[17]:


df.corr()


# In[21]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# In[ ]:




