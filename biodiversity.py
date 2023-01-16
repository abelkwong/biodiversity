#!/usr/bin/env python
# coding: utf-8

# # A Closer Look at Biodiversity Across National Parks In The USA

# ## Introduction

# Biodiversity is a term used to describe the vast collection of life and its variabilities here on planet Earth. For this project, biodiversity data obtained from the National Parks Service will be analyzed in order to learn more about endangered species in various national parks. More specifically, data analysis will be performed on the conservation statuses of the species to find out if there are any patterns or themes present among the endangered. 
# 
# During this project, the data will be processed, analyzed, and visually plotted to help answer the following questions posed:
# - What is the distribution of conservation status?
# - Are certain types of animal categories more likely to be endangered?
# - Are differences between species and conservation statuses significant?
# - Which park contains the highest population of endangered species?
# 
# **<u>Source:</u>**
# 
# *All data has been gracefully provided by [Codecademy.com](www.codecademy.com) for the purpose of this project. Note that the data is inspired by real life applications but is mostly fictional.*

# ## Import Python Modules

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load Data

# In[2]:


# load the csv datas
df_species = pd.read_csv('species_info.csv')
df_obs = pd.read_csv('observations.csv')


# **<u>Species Information</u>**
# 
# The number of columns in `df_species` is four and the names of these columns are: **category**, **scientific_name**, **common_names**, and **conservation_status**. 

# In[3]:


print(df_species.columns)
df_species.head()


# **<u>Observations</u>**
# 
# The number of columns in `df_obs` is three and the names of these columns are: **scientific_name**, **park_name**, and **observations**.

# In[4]:


print(df_obs.columns)
df_obs.head()


# ## Explore Data

# **<u>Species Information</u>**
# 
# There are a total of 5842 entries for the first dataset. 

# In[5]:


# preview of dataset
print(df_species.info())


# In the `category` column, it would be useful to find the number and names of animal categories present in the dataset. 
# 
# To obtain the results, let us use the following methods:

# In[6]:


# Count number of unique categories
print(df_species['category'].nunique())


# In[7]:


# Display names of unique categories
print(df_species['category'].unique())


# Using the pandas module, the total number of animal categories is 7 and the animals are separated into the following categories: **Mammal, Bird, Reptile, Amphibian, Fish, Vascular Plant,** and **Nonvascular Plant.**

# **<u>Observations</u>**

# In the observations dataset, there is a column designated for the various national parks where the animals were observed. To find the names of the parks listed, the following method was used. The results show four unique parks with the following names: **Great Smoky Mountains National Park, Yosemite National Park, Bryce National Park,** and **Yellowstone National Park**.

# In[8]:


# Count number of unique parks
print(df_obs['park_name'].unique())


# Next, the total number of observations was obtained by summing all of values in the observation column. The total number of observations was 3314739.

# In[9]:


# Count total number of observations
print(df_obs['observations'].sum())


# ## Analysis

# **Distribution of Conservation Status**

# To find the distribution of the conservation status, we can first examine the `conservation_status` column more closely. A frequency table was created in order to find the unique names and counts for each conservation status. The statuses found include: 
# - Species of concern
# - Endangered 
# - Threatened
# - In recovery
# - Nan

# In[10]:


# Find unique values in conservation_status column
df_species['conservation_status'].value_counts(dropna=False)


# Notice the significantly large number of null values that were found in the dataset, there were 5633 NaN values! This will impact the distribution of `conservation_status` so the null values will be excluded. In addition, to provide consistency with the naming of the statuses, the 'NaN' value was replaced with 'No Contact'. 

# In[11]:


# Fill all null values with 'No Contact' status
df_species['conservation_status'].fillna('No Contact', inplace=True)
df_species.head(10)


# Now, the `conservation_status` column is preprocessed for all values including null values and ready for visualization. Since the values found inside `conservation status` are categorical, a histogram will be used to plot the frequency of each status. As stated previously, due to the high frequency of 'No Contact' statuses, the histogram will exclude all null values. In order to do this, we will first create a new variable excluding the `"No Contact"` values.
# 
# According to the diagram, the histogram is skewed to the right, meaning that most data can be found on the left-hand side. Most observations can be classified as `"Species of Concern"`, which can provide experts with species for concentration to prevent further endangerment risks.

# In[12]:


# create category variable for plot
conservation_statuses = df_species[df_species['conservation_status'] != 'No Contact']


# In[13]:


# plot histogram of conservationCategories
plt.figure(figsize=(8, 6))

plt.hist(conservation_statuses['conservation_status'])
plt.title("Distribution of Conservation Statuses")
plt.show()
plt.clf()


# **Bivariate Analysis of Animal Categories and Conservation Status**

# Next, we would like to examine if certain types of animal categories are more likely to be endangered. In order to do this, we will study the relationship between animal categories and conservation status. First, the proportions of animal categories for each conservation status was discovered.

# In[14]:


# Examine proportions of animal categories for each conservation status
conservation_proportions = df_species.groupby(['conservation_status','category'])['common_names'].count().unstack()
conservation_proportions_v2 = conservation_proportions.reindex    (index=['No Contact','In Recovery','Species of Concern','Threatened','Endangered'],    columns=['Amphibian','Bird','Fish','Mammal','Reptile','Nonvascular Plant','Vascular Plant'])

conservation_proportions_v2


# Typically, the values alone are not the most useful for statistical interpretation so instead, the values will be transformed into percentages of a whole. In addition, due to the large population of `"No Contact"` statuses, the percentages table will exclude this data. To prevent errors during transformation, the `NaN` values will be replaced with zeroes.

# In[15]:


# Replace NaN values with zero
conservation_proportions_v2.fillna(0, inplace=True)

# Remove "No Contact" index
conservation_proportions_v2 = conservation_proportions_v2.drop(index="No Contact")

# Transform proportions table to percentages
conservation_proportions_v2 = conservation_proportions_v2.div(conservation_proportions_v2.sum(axis=1), axis=0)
conservation_proportions_v2 = round(conservation_proportions_v2.mul(100), 2)
conservation_proportions_v2['Total Percentage'] = conservation_proportions_v2.sum(axis=1)

conservation_proportions_v2


# After conversion, all values listed in the table are in percentage form. In addition, `Total Percentage` was a column added that summed all of the values in each row to confirm that the percentages all add up to 1. The data is now ready for visualization, and, for this purpose, modified pie charts (also known as donut charts) will be used.

# In[16]:


# Drop 'Total Percentage' 
conservation_proportions_v3 = conservation_proportions_v2.drop(columns='Total Percentage')

# Donut charts
# pop values
conservation_labels = ['In Recovery', 'Species of Concern', 'Threatened', 'Endangered']
colours = ['#4c9cbc','#69a6bb','#0e4866','#9d6a4b','#a7c2c6','#233c44','#aaaeb7']

fig, axes = plt.subplots(2,2, figsize=(18,18))

axes[0,0].pie(conservation_proportions_v3.loc['In Recovery'],
        colors = colours,
        wedgeprops = {'width': 0.25, 'linewidth': 1, 'edgecolor': 'white'}, 
        labels = ['','Bird','','Mammal','','',''], 
        autopct = (lambda pct:'{:1.1f}%'.format(pct) if pct > 0 else ''), 
        textprops = {'fontsize':12})
                        
axes[0,1].pie(conservation_proportions_v3.loc['Species of Concern'],
        colors = colours,
        wedgeprops = {'width': 0.25, 'linewidth': 1, 'edgecolor': 'white'}, 
        labels = ['Amphibian','Bird','Fish','Mammal','Reptile','Nonvascular Plant','Vascular Plant'],
        autopct = (lambda pct:'{:1.1f}%'.format(pct) if pct > 0 else ''),
        textprops = {'fontsize':12})

axes[1,0].pie(conservation_proportions_v3.loc['Threatened'],
        colors = colours,
        wedgeprops = {'width': 0.25, 'linewidth': 1, 'edgecolor': 'white'},
        labels = ['Amphibian','','Fish','Mammal','','','Vascular Plant'],
        autopct = (lambda pct:'{:1.1f}%'.format(pct) if pct > 0 else ''),
        textprops = {'fontsize':12})

axes[1,1].pie(conservation_proportions_v3.loc['Endangered'],
        colors = colours,
        wedgeprops = {'width': 0.25, 'linewidth': 1, 'edgecolor': 'white'},
        labels = ['Amphibian','Bird','Fish','Mammal','','','Vascular Plant'],
        autopct = (lambda pct:'{:1.1f}%'.format(pct) if pct > 0 else ''),
        textprops = {'fontsize':12})

axes[0,0].set_title(conservation_labels[0], fontweight='bold', size=20)
axes[0,1].set_title(conservation_labels[1], fontweight='bold', size=20)
axes[1,0].set_title(conservation_labels[2], fontweight='bold', size=20)
axes[1,1].set_title(conservation_labels[3], fontweight='bold', size=20)


# There were only two categories in recovery: birds and mammals. While all species were of concern, the most dominant ones were birds, vascular plant, and mammals with percentages greater than 15%. Four species were found to be threatened and posed the risk of endangerment. The highest population for threatened species was fish while mammals, amphibians and vascular plants displayed an equal percentage. Lastly, the three main species considered endangered include mammals, birds, and fish with percentages greater than 18%. It is concerning to observe that almost half of the endangered species were mammals, while a quarter of the endangered population were birds. 

# **Statistical Significance**

# In this section, we will use chi-squared tests in order to analyze if there is any statistical significance between the species and rates of conservation status. In order to calculate the rate of conservation status, we will first separate any species that had a status other than `No Contact` in a new column called `is_protected`. 

# In[50]:


df_species['is_protected'] = df_species['conservation_status'] != 'No Contact'


# Next, a dataframe will be created that shows the different animal categories that are protected and not protected. 

# In[51]:


protected_categories = df_species.groupby(['category', 'is_protected']).scientific_name.nunique().reset_index().pivot(columns='is_protected', index='category', values='scientific_name').reset_index()
protected_categories.columns = ['category', 'not_protected', 'protected']

protected_categories['protected_rate'] = protected_categories['protected'] / (protected_categories['protected'] + protected_categories['not_protected']) * 100

protected_categories


# Once we have the protected rates, we can now run the chi square tests for all species categories in pairs.

# In[ ]:





# **Park Analysis**

# In[ ]:





# ## Conclusion

# ## Additional Notes

# In[ ]:




