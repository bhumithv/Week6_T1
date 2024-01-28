#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[6]:


data = pd.read_csv("E:\Data Analytics\week6\Indian Cars-20240128T064735Z-001\Indian Cars\cars_ds_final.csv")


# In[7]:


data.head()


# In[8]:


data.tail()


# In[11]:


data = data.copy()


# In[12]:


data.columns=data.columns.str.lower()


# In[13]:


data.columns


# In[14]:


data.isna().sum()


# # Getting information about the dataset
# 

# In[16]:


data.info(verbose=True)


# # Getting shape of the dataset
# 

# In[17]:


data.shape


# # Getting description of the dataset

# In[18]:


data.describe()


# # Getting a random sample from the dataset 

# In[19]:


data.sample()


# # DATA CLEANING
# 

# Dropping unnecessary column
# 

# In[20]:


data = data.drop('unnamed: 0', axis=1)


# Dropping rows from the 'make' column which has null values
# 

# In[21]:


data.dropna(subset=['make'], inplace=True)


# Getting a random sample from the dataset

# In[22]:


data.sample()


# Performing string formatting

# In[23]:


data['ex-showroom_price'] = data['ex-showroom_price'].str.replace('Rs. ', '')

data['ex-showroom_price'] = data['ex-showroom_price'].str.replace(',', '')

data['ex-showroom_price'] = data['ex-showroom_price'].astype(int)

data.sample()
     


# Getting the count of the values present in 'drivetrain' column

# In[24]:


data['drivetrain'].value_counts()


# Dropping rows from the 'displacement' column which has null values
# 

# In[26]:


data.dropna(subset=['displacement'], inplace=True)


# Dropping rows from the 'drivetrain' column which has null values

# In[27]:


data.dropna(subset=['drivetrain'], inplace=True)


# Checking for number of null values in 'drivetrain' column

# In[28]:


data['drivetrain'].isna().sum()


# Dropping rows from the 'drivetrain' column which has null values

# In[30]:


data.dropna(subset=['drivetrain'], inplace=True)


# Checking for number of null values in 'cylinder_configuration' column

# In[31]:


data['cylinder_configuration'].isna().sum()


# Getting the count of the values present in 'cylinder_configuration' column

# In[33]:


data['cylinder_configuration'].value_counts()


# Dropping rows from the 'cylinder_configuration' column which has null values

# In[35]:


data.dropna(subset=['cylinder_configuration'], inplace=True)


# Getting the count of the values present in 'emission_norm' column

# In[37]:


data['emission_norm'].value_counts()


# Checking for number of null values in 'emission_norm' column

# In[38]:


data['emission_norm'].isna().sum()
     


# Dropping rows from the 'emission_norm' column which has null values

# In[39]:


data.dropna(subset=['emission_norm'], inplace=True)


# Getting the names of the columns and putting them into a list if 'Yes' is present in them

# In[40]:


columns_with_yes = data.columns[data.eq('Yes').any()]

columns_with_yes_list = columns_with_yes.tolist()

print("Columns with 'Yes':", columns_with_yes_list)


# Filing the null values to the columns_with_yes with 'No' in place of null (NaN) value

# In[41]:


data[columns_with_yes_list] = data[columns_with_yes_list].fillna('No')
     


# Getting the names of the columns which is null and of object datatype (Exluding the numeric columns)

# In[42]:


object_columns_with_nan = data.columns[(data.isnull().any()) & (data.dtypes == 'object')]

object_columns_with_nan_list = object_columns_with_nan.tolist()

print("Object columns with NaN values:", object_columns_with_nan_list)


# Filing the null values to the object_columns_with_nan with 'Not Available' in place of null (NaN) value

# In[43]:


replace_values = {column: 'Not Available' for column in object_columns_with_nan}

data.fillna(replace_values, inplace=True)


# Getting information about the dataset

# In[44]:


data.info(verbose=True)


# Filling the null values of 'usb_ports' columns with 0

# In[45]:


data['usb_ports'].fillna(0, inplace=True)


# Getting description about the dataset

# In[46]:


data.describe()


# Getting 5 random samples from the dataset

# In[47]:


data.sample(5)
     


# # DATA VISUALIZATION

# Visualizing distribution of Ex-Showroom price

# In[52]:


plt.figure(figsize=(10, 6))
ax = sns.histplot(y=data['ex-showroom_price'], bins=30, kde=False)
plt.title('Distribution of Ex-Showroom Price')
plt.xlabel('Frequency')
plt.ylabel('Ex-Showroom Price')

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.show()


# Visualizing the count of Car Manufacturers

# In[53]:


plt.figure(figsize=(12, 6))
ax = sns.countplot(x='make', data=data)
plt.title('Count of Car Manufacturers')
plt.xlabel('Car Manufacturers')
plt.ylabel('Count')
plt.xticks(rotation=90)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()


# Visualizing the Manufacturers and their respective Models via sunburst chart

# In[54]:


fig = px.sunburst(data, path=['make', 'model'], title=' Models and Manufacturers Sunburst Chart', hover_name = 'model')

fig.show()


# Visualizing Box Plot of all numeric columns

# In[56]:


numeric_columns = data.select_dtypes(include='number').columns

plt.figure(figsize=(16, 8))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(1, len(numeric_columns), i)
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot of {column}', fontsize = 7.5, fontweight = 'bold')

plt.show()


# Visualizing Correlation Matrix

# In[57]:


numeric_columns = data.select_dtypes(include='number')

correlation_matrix = numeric_columns.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Variables')
plt.show()


# Visualizing the scatter plot of Seating Capacity vs Number of Airbags with Ex-Showroom Price

# In[59]:


plt.figure(figsize=(12, 8))
plt.scatter(data['seating_capacity'], data['number_of_airbags'], c=data['ex-showroom_price'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Ex-Showroom Price')
plt.xlabel('Seating Capacity')
plt.ylabel('Number of Airbags')
plt.title('Scatter Plot: Seating Capacity vs Number of Airbags with Ex-Showroom Price')
plt.show()


# Visualizing pair-plot for numeric data

# In[60]:


numeric_columns = data.select_dtypes(include='number')

sns.pairplot(numeric_columns)
plt.suptitle('Pair Plot of Numeric Variables', y=1.02)
plt.show()


# Visualizing the count of Car Models with Manufacturers

# In[66]:


plt.figure(figsize=(12, 6))
sns.countplot(x='model', hue='make', data=data)
plt.title('Count of Car Models with Manufacturers')
plt.xlabel('Car Models')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Manufacturers', loc='upper right')
plt.show()


# Visualzing the joint Plot of Seating Capacity vs Ex-Showroom Price

# In[67]:


sns.jointplot(x='seating_capacity', y='ex-showroom_price', data=data, kind='scatter')
plt.suptitle('Joint Plot: Seating Capacity vs Ex-Showroom Price', y=1.02)
plt.show()
     


# Visualizing the box Plot of Ex-Showroom Price by Manufacturers with Car Models

# In[68]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='make', y='ex-showroom_price', hue='model', data=data)
plt.title('Box Plot: Ex-Showroom Price by Manufacturers with Car Models')
plt.xlabel('Manufacturers')
plt.ylabel('Ex-Showroom Price')
plt.xticks(rotation=90)
plt.legend(title='Car Models', loc='best')
plt.show()


# Visualizing the categorical Plot of Ex-Showroom Price by Car Model

# In[69]:


plt.figure(figsize=(12, 8))
sns.catplot(x='ex-showroom_price', y='make', data=data, kind='box')
plt.title('Categorical Plot: Ex-Showroom Price by Car Model')
plt.ylabel('Car Models')
plt.xlabel('Ex-Showroom Price')
plt.show()


# Visualizing the violin plot for numeric columns

# In[71]:


numeric_columns = data.select_dtypes(include='number')

plt.figure(figsize=(16, 8))
for i, column in enumerate(numeric_columns.columns, 1):
    plt.subplot(1, len(numeric_columns.columns), i)
    sns.violinplot(x=column, data=data)
    plt.title(f'Violin Plot: {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()

