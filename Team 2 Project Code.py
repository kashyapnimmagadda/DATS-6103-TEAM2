# DATS-6103-11 Group Project - Team 2
# Spring 2023
# Group members: Muhannad Alwhebie, Brian Gulko, Mengfei Hung, Kashyap Nimmagadda
# The George Washington University - Data Science Program

"""
Description:
Our project uses data from Open Data DC and describes the sale history for active properties listed among the District of Columbia's real property tax assessment roll. The dataset contains about 108,000 rows and 39 columns describing property attributes such as area and number of bedrooms as well as sale information such as sale price and date. Our goal is to use analysis and models to better understand the relationship between these attributes and the effects they have on sale price.
"""

"""
SMART Questions (from our topic proposal, may be modified as we go):
1.	What are the characteristics of an average residential property?
    - Which heating type is the most common in residential properties in this dataset, and what is the percentage of properties with this heating type?
    - What is the average number of bathrooms and half-bathrooms in residential properties in this dataset?
    - What is the average land area of residential properties in this dataset, and how does this vary by number of bedrooms?
    - How has the gross building area of residential properties in this dataset changed over time?

2.	Which variables have an impact on sale price, and how strong is that impact?
    - Is there a correlation between the number of bedrooms and the sale price of a residential property in this dataset?
    - Is there a correlation between the grade and the sale price of a residential property in this dataset?
    - Is there a correlation between gross building area and sale price?

3.	Did COVID-19 have an impact on residential sale prices? If so how big was that impact?
    - How did the average grade of residential properties sold in the District of Columbia change during the COVID-19 pandemic, and was this related to changes in sale price?

Instructor's feedback: 
    - Nice job in setting up the descriptive and inferential statistical questions and subquestions separately.
    - In addition to questions related to association of specific attributes and sale price, you may consider a question related to feature selection and ranking. In other words, what attributes are most predictive of sales price after adjustment of other related attributes?
    - For the third question, please clarify what role grade of the property has in the impact analysis. Is it tested as a mediator or moderator of the association of pandemic and sale price? Do we need to adjust for inflation when you compare the pre- and post comparison?

"""

"""
Links
Dataset Source: This data comes from Open Data DC's Computer Assisted Mass Appraisal - Residential dataset which can be found at this link:
https://opendata.dc.gov/datasets/DCGIS::computer-assisted-mass-appraisal-residential/explore

Attribute information and descriptions can be found at this link:
https://www.arcgis.com/sharing/rest/content/items/c5fb3fbe4c694a59a6eef7bf5f8bc49a/info/metadata/metadata.xml?format=default&output=html

GitHub Repo: https://github.com/kashyapnimmagadda/DATS-6103-TEAM2.git
"""

"""
Code Outline (current plan, may change):
1. Setup
2. Prepricessing
3. EDA
4. Descriptive Characteristics
5. Modeling Sales Price (linear and multiple regression, maybe others)
6. Modeling if the property sold for a price (logistic regression, classification tree, maybe others)
7. COVID Comparison
"""

"""
Data Dictionary:
To add using descriptions from https://www.arcgis.com/sharing/rest/content/items/c5fb3fbe4c694a59a6eef7bf5f8bc49a/info/metadata/metadata.xml?format=default&output=html
"""

"""
Notes:
- We will use 2010 through 2022 for our analysis.
- For sale price, filter out properties that sold for $0.
- Linear and multiple regression for sale price for homes that sold for more than $0
- Logistic regression for if the property sold for a price
"""

### Setup
#%%
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Import Data
sales = pd.read_csv("Computer_Assisted_Mass_Appraisal_-_Residential_3-24-23.csv")

#%%
# Check data
print(sales.shape)
print(sales.info())
print(sales.head())


### Preprocessing


# Sales per year

# Sales per year with price vs no price


# Number of years/months/days since 1/1/2010

# Other munging

# Removing unneeded variables



### EDA

### Descriptive Characteristics

### Modeling Sales Price (linear and multiple regression)

### Modeling if the property sold for a price (logistic regression)

### COVID Comparison

