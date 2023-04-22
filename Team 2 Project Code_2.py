#%%[markdown]
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
2. Preprocessing
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
sales = pd.read_csv("Computer_Assisted_Mass_Appraisal_-_Residential_3-24-23.csv", parse_dates=["SALEDATE"])

# Rename columns to lowercase for convenience
sales.columns= sales.columns.str.lower()
#%%
# Check data
print(sales.shape)
print(sales.info())
print(sales.head())


### Preprocessing
## Creating new variables
#%%
# Create new variables for the year, month, and day to work with
sales["sale_year"] = sales["saledate"].dt.year
sales["sale_month"] = sales["saledate"].dt.month
sales["sale_day"] = sales["saledate"].dt.day
print(sales["sale_day"])
#%%
# Create a variable for the month name
month_names = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")

sales["sale_named_month"] = sales["sale_month"].map(lambda x: month_names[x-1])

# Convert to an ordered categorical variable
sales["sale_named_month"] = pd.Categorical(sales["sale_named_month"], categories = month_names, ordered = True)

sales["sale_named_month"].head()

#%%
# Create a boolean variable for if the property sold for a price over $)
sales.insert(sales.columns.get_loc("price")+1, "with_price", sales["price"].apply(lambda x: False if x == 0 else True))

# Create a boolean for if the property was remodeled
sales.insert(sales.columns.get_loc("yr_rmdl")+1, "remodeled", sales["yr_rmdl"].notnull())

#%%
# Create variables looking at how much time has passed since January 1 2010, the start date of the subset we are working with
def year_diff(a, b):
    return (a.dt.year - b.year)

def month_diff(a, b):
    return 12 * (a.dt.year - b.year) + (a.dt.month - b.month)

sales["num_years_passed"] = year_diff(sales["saledate"], pd.Timestamp("2010/01/01 00:00:00+00"))
sales["num_months_passed"] = month_diff(sales["saledate"], pd.Timestamp("2010/01/01 00:00:00+00"))
sales["num_days_passed"] = (sales["saledate"] - pd.Timestamp("2010/01/01 00:00:00+00")).dt.days


## Sales per year
#%%
print(sales["sale_year"].value_counts())

#%%
# Bar chart of sales per year
year_graph_data = sales["sale_year"].value_counts().rename_axis(["sale_year"]).reset_index(name = "count")

year_graph = sns.barplot(data = year_graph_data, x = "sale_year", y = "count", color = "steelblue")
year_graph.set(title = "Number of Sales by Year", xlabel = "Year")
year_graph.set_xticks(range(0, 69, 5))
plt.show()

## Other preprocessing (not sure what else is needed)


#%%
## Removing unneeded variables and filtering
# There are 20 rows in our subset that are missing the sale price. Given the imoprtance of this variable in our project and the small number of sales missing this data, it makes sense to filter out these rows

# Number of rows in our subset missing sale price
print(len(sales[(sales["sale_year"] >= 2010) & (sales["sale_year"] < 2023) & sales["price"].isnull()]))

# We can add more variables this if we identify more unneeded variables
cols_to_drop = ["ssl", "gis_last_mod_dttm", "objectid"]

# We will use this dataframe for the rest of the analysis
sales_trimmed = sales[(sales["sale_year"] >= 2010) & (sales["sale_year"] < 2023) & sales["price"].notnull()].drop(cols_to_drop, axis = 1)



#%%
### EDA

#%%
print(sales_trimmed.shape)
print(sales_trimmed.info())
# For this it makes sense to only use sales where the property sold for a price, not for $0
sales_trimmed_with_price = sales_trimmed[sales_trimmed["with_price"] == True]

#%%

fig, axs = plt.subplots(ncols=2, figsize=(15,5))

# Histogram of the number of properties sold by year
axs[0].hist(sales["sale_year"], bins=range(2010,2024), edgecolor='black')
axs[0].set_xlabel("Year")
axs[0].set_ylabel("Number of Properties Sold")
axs[0].set_title("Number of Properties Sold by Year")

# Histogram of the number of properties sold by month
axs[1].hist(sales["sale_month"], bins=range(1,13), edgecolor='black')
axs[1].set_xlabel("Month")
axs[1].set_ylabel("Number of Properties Sold")
axs[1].set_title("Number of Properties Sold by Month")

plt.show()


price_counts = sales["with_price"].value_counts()

# Bar chart of the sales by price
plt.bar(["With Price", "Without Price"], price_counts.values)
plt.xlabel("Price")
plt.ylabel("Number of Sales")
plt.title("Sales by Price")
plt.show()

# Total Count of number of sales with and without a remodel
remodel_counts = sales["remodeled"].value_counts()

# Bar chart of the sales by remodel status
plt.bar(["Remodeled", "Not Remodeled"], remodel_counts.values)
plt.xlabel("Remodel Status")
plt.ylabel("Number of Sales")
plt.title("Sales by Remodel Status")
plt.show()
# Sales per year with price vs no price stacked bar chart``

#change to years 2010 throough 2021
sales_subset = sales[(sales["sale_year"] >= 2010) & (sales["sale_year"] < 2023)]
sales_with_price = sales_subset.groupby(["sale_year", "with_price"]).size().unstack(fill_value=0)
sales_with_price.plot(kind="bar", stacked=True)
plt.xlabel("Year")
plt.ylabel("Number of Sales")
plt.title("Sales Per Year With/Without Price")
plt.show()

#%%
plt.figure(figsize=(10,6))
sns.histplot(sales_trimmed_with_price['price'], kde=True)
plt.xlim(0, 0.5e7)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.show()

#%%
median_prices_by_year = sales_trimmed_with_price.groupby("sale_year")["price"].median().reset_index(name="median_price")

sns.lineplot(x="sale_year", y="median_price", data=median_prices_by_year)
plt.xlabel("Sale Year")
plt.ylabel("Median Sale Price")
plt.title("Median Sale Price by Year")
plt.show()

#%%
# Computing the correlation matrix
corr = sales_trimmed.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generating a custom diverging colormap
cmap = sns.diverging_palette(10, 220, sep=80, n=7)

# Heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#%%
print(corr)

#%%
### Descriptive Characteristics

summary_stats = sales_trimmed.describe()
print(summary_stats)

#%%
# Creating scatterplots of highly correlated variables
sns.boxplot(x='bathrm', y='price', data=sales_trimmed_with_price)
plt.title('Bathrooms vs. Price')
plt.show()

sns.boxplot(x='rooms', y='price', data=sales_trimmed_with_price)
plt.title('Rooms vs. Price')
plt.show()

sns.boxplot(x='bedrm', y='price', data=sales_trimmed_with_price)
plt.title('Bedrooms vs. Price')
plt.show()

sns.scatterplot(x='gba', y='price', data=sales_trimmed_with_price)
plt.title('Gross Building Area vs. Price')
plt.show()

sns.boxplot(x='kitchens', y='price', data=sales_trimmed)
plt.title('Kitchens vs. Price')
plt.show()

# Filter the data based on Year Built range
df_filtered = sales_trimmed_with_price[(sales_trimmed_with_price["ayb"] >= 2010) & (sales_trimmed_with_price["ayb"] <= 2023)]

plt.scatter(df_filtered["ayb"], df_filtered["price"])
plt.xlabel("Year Built")
plt.ylabel("Price")
plt.title("Scatter plot of Price vs Year Built (2010-2023)")
plt.show()

# Scatter plot of num_units vs price
plt.scatter(sales_trimmed_with_price['num_units'], sales_trimmed_with_price['price'])
plt.xlabel('Number of Units')
plt.ylabel('Price')
plt.show()
#%%
sns.boxplot(x="cndtn", y="price", data=sales_trimmed_with_price)
plt.title("Sale Price by Condition")
plt.xlabel("Condition")
plt.ylabel("Sale Price")
plt.show()
#%%
import seaborn as sns

sns.countplot(x='cndtn', data=sales_trimmed_with_price)
plt.xlabel('Condition')
plt.ylabel('Number of Properties')
plt.title('Number of Properties per Condition Category')
plt.show()

#%%


#%%

bathroom_freq = sales_trimmed_with_price["bathrm"].value_counts().sort_index()
plt.bar(bathroom_freq.index, bathroom_freq.values, edgecolor='black')
plt.xlabel("Number of Bathrooms")
plt.ylabel("Frequency")
plt.title("Bar graph of Number of Bathrooms")
plt.show()


#Not useful
# Bar plot of extwall
plt.bar(sales_trimmed_with_price['extwall'], sales_trimmed_with_price['price'])
plt.xlabel('Exterior Wall Type')
plt.ylabel('Price')
plt.show()

# %%
### Building LR Model as price as indep to addrees the SMART Q
# Import required libraries

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Drop rows with missing values
sales_trimmed_with_price = sales_trimmed_with_price.dropna()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sales_trimmed_with_price[['bathrm', 'bedrm', 'grade']], sales_trimmed_with_price['price'], test_size=0.2, random_state=42)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R-squared:", r2)


##Interp: The output shows the Mean Squared Error (MSE) and R-squared values of the linear regression model.

#The mean squared error measures the average of the squared differences between the predicted values and actual values of the target variable. The MSE value of 2.2  (approximately) means that, on average, the model's predictions are off by about 14.9 million dollars from the actual price of the property.

#The R-squared value is a statistical measure that represents the proportion of variance in the target variable that is explained by the predictor variables. In this case, the R-squared value of 0.58 means that the model explains 58% of the variance in the target variable, while the remaining 42% of the variance is unexplained by the model. This indicates that the model may not be capturing all the important factors that determine the price of a property.

# %%
## addin heat as predictor 


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sales_trimmed_with_price[['bathrm', 'bedrm', 'grade', 'heat']], sales_trimmed_with_price['price'], test_size=0.2, random_state=42)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R-squared:", r2)
##The new output shows that adding the 'heat' variable did not have a significant impact on the model's performance as the Mean Squared Error (MSE) and R-squared values are almost similar to the previous model's output. The MSE value of 2.22  (approximately) means that, on average, the model's predictions are off by about 14.9 million dollars from the actual price of the property. The R-squared value of 0.58 means that the model explains 58.4% of the variance in the target variable, while the remaining 41.6% of the variance is unexplained by the model. This indicates that there may be other important factors that influence the price of a property that the model has not captured.
# %%
#updated code that includes heat and cndtn as additional predictor variables:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    sales_trimmed_with_price[['bathrm', 'bedrm', 'grade', 'heat', 'cndtn']],
    sales_trimmed_with_price['price'],
    test_size=0.2,
    random_state=42
)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R-squared:", r2)

# Print the coefficients of the model
print("Coefficients:", model.coef_)
#Based on the output, we can see that the model with the additional predictor variables (heat and cndtn) has a lower mean squared error and a higher R-squared value compared to the model with only bathrm, bedrm, and grade. This suggests that the inclusion of heat and cndtn as predictor variables has improved the model's ability to predict the sale price of residential properties.

#To answer the specific questions:

#Is there a correlation between the number of bedrooms and the sale price of a residential property in this dataset?
#The coefficient for bedrm in the model is 13,941. This means that, all other variables being equal, a one-unit increase in the number of bedrooms is associated with a $13,941 increase in the sale price of the property. This indicates that there is a positive correlation between the number of bedrooms and the sale price of a residential property in this dataset.

#Is there a correlation between the grade and the sale price of a residential property in this dataset?
#The coefficient for grade in the model is 303,631. This means that, all other variables being equal, a one-unit increase in the grade is associated with a $303,631 increase in the sale price of the property. This indicates that there is a strong positive correlation between the grade and the sale price of a residential property in this dataset.

#Is there a correlation between gross building area and sale price?
#Gross building area is not included as a predictor variable in the model. Therefore, we cannot determine the correlation between gross building area and sale price based on this model.
# %%
# Adding  gross building area another predictor 


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    sales_trimmed_with_price[['bathrm', 'bedrm', 'grade', 'heat', 'cndtn', 'gba']],
    sales_trimmed_with_price['price'],
    test_size=0.2,
    random_state=42
)

# Instantiate the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R-squared:", r2)

# Print the coefficients of the model
print("Coefficients:", model.coef_)

#The updated model includes the "GBA" variable as an additional predictor variable to investigate its impact on sale price. The output shows an improved performance of the model with a lower mean squared error of approximately 1.64 and a higher R-squared value of approximately 0.69. This indicates that the addition of the "GBA" variable has improved the model's ability to explain the variance in the target variable.

#The coefficients of the model suggest that the "GBA" variable has a positive impact on the sale price, with a coefficient value of 250701.29. This means that, on average, for every one-unit increase in the gross building area, the sale price of the property is expected to increase by approximately $250,701.



# %%
# Gradient Boosting Classifier model for property was sold for more than $500,000 or not

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Define the predictor variables
X = sales_trimmed_with_price[['bathrm', 'bedrm', 'gba', 'cndtn']]

# Define the threshold
threshold = 500000 

# Define the binary target variable
y = (sales_trimmed_with_price['price'] > threshold).astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Gradient Boosting Classifier model
model = GradientBoostingClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Plot the confusion matrix
plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for Gradient Boosting Classifier Model')
plt.show()

# Prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Instantiate the Gradient Boosting Classifier model
model = GradientBoostingClassifier()

# Evaluate the model using cross-validation
scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)

# Report performance
print('Accuracy: %.3f (%.3f)' % (scores.mean(), scores.std()))

#The output shows the performance of the Gradient Boosting Classifier model for predicting whether a property was sold for more than $500,000 or not.

#The model achieved an accuracy of 0.812, which means that it correctly predicted 81.2% of the test cases. It also achieved a precision of 0.841, which means that out of all the properties that the model predicted were sold for more than $500,000, 84.1% of them were actually sold for that price or more. The recall of the model was 0.925, which means that out of all the properties that were actually sold for more than $500,000, the model correctly identified 92.5% of them.

#The confusion matrix shows that the model predicted 173 properties as sold for more than $500,000, out of which 146 were actually sold for that price or more. The model also predicted 51 properties as sold for less than $500,000, out of which 40 were actually sold for less than that price.

#The cross-validation score of the model was 0.806, which means that the model's performance was consistent across the different folds of the data.








# %%
### COVID Comparison
# Split to 3 periods(Ex Ante, lockdown, Ex Post)
ExAnte = sales_trimmed[(sales_trimmed["sale_year"] >= 2010) & (sales_trimmed["sale_year"] < 2020)]
lockdown = sales_trimmed[(sales_trimmed["sale_year"] >= 2020) & (sales_trimmed["sale_year"] < 2021)]
ExPost = sales_trimmed[(sales_trimmed["sale_year"] >= 2021) ]

#sales_trimmed['sale_year'] = pd.to_datetime(sales_trimmed['sale_year'])
sales["sale_year"] = sales["saledate"].dt.year

sales_trimmed = sales[(sales["sale_year"] >= 2018) & (sales["sale_year"] <= 2023) & sales["price"].notnull()].drop(cols_to_drop, axis = 1)
#bins = [pd.to_datetime('2019-01-01'), pd.to_datetime('2020-01-01'), pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')]
sales_trimmed['period'] = pd.cut(sales_trimmed['sale_year'], bins=[2018,2020,2022,2023], labels=['ExAnte', 'Lock down', 'ExPost'])
print(sales_trimmed['period'])

sales_num_trimmed = sales[(sales["sale_year"] >= 2018) & (sales["sale_year"] <= 2023) & sales["sale_num"].notnull()].drop(cols_to_drop, axis = 1)
sales_num_trimmed['period'] = pd.cut(sales_num_trimmed['sale_year'], bins=[2018,2020,2021,2023], labels=['ExAnte', 'Lock down', 'ExPost'])
print(sales_num_trimmed['period'])



# %%
## Boxplot of residential sale prices in 3 periods(Ex Ante, lockdown, Ex Post)

sns.boxplot(x="period", y="price", data=sales_trimmed)
plt.title("Sale Price by period")
plt.xlabel("Period")
plt.ylabel("Sale Price")
plt.show()
# %%

sns.boxplot(x="bathrm", y="price",hue="period", data=sales_trimmed, palette="pastel")
plt.title("Sale Price by period")
plt.xlabel("Number of bethroom")
plt.ylabel("Sale Price")
plt.show()
# %%
sns.boxplot(x="bedrm", y="price",hue="period", palette="pastel", data=sales_trimmed)
plt.title("Sale Price by period")
plt.xlabel("Number of bedroom")
plt.ylabel("Sale Price")
plt.show()
# %%
sns.boxplot(x="gba", y="price",hue="period", data=sales_trimmed)
plt.title("Sale Price by Number of gba")
plt.xlabel("gba")
plt.ylabel("Sale Price")
plt.show()
# %%
sns.boxplot(x="cndtn", y="price",hue="period", data=sales_trimmed)
plt.title("Sale Price by  Condition of the House")
plt.xlabel("cndtn")
plt.ylabel("Sale Price")
plt.show()
# %%



## Boxplot of residential sale number in 3 periods(Ex Ante, lockdown, Ex Post)

sns.boxplot(x="period", y="sale_num", data=sales_num_trimmed)
plt.title("Sale Number by period")
plt.xlabel("period")
plt.ylabel("Sale Number")
plt.show()
# %%
sns.boxplot(x="bathrm", y="sale_num",hue="period", data=sales_num_trimmed)
plt.title("Sale Price by period")
plt.xlabel("Number of bethroom")
plt.ylabel("Sale Price")
plt.show()
# %%
sns.boxplot(x="bedrm", y="sale_num",hue="period", data=sales_num_trimmed)
plt.title("Sale Price by period")
plt.xlabel("Number of bedroom")
plt.ylabel("Sale Price")
plt.show()
# %%
sns.boxplot(x="gba", y="sale_num",hue="period", data=sales_num_trimmed)
plt.title("Sale Price by Number of gba")
plt.xlabel("gba")
plt.ylabel("Sale Price")
plt.show()
# %%
sns.boxplot(x="cndtn", y="sale_num",hue="period", data=sales_num_trimmed)
plt.title("Sale Price by  Condition of the House")
plt.xlabel("cndtn")
plt.ylabel("Sale Price")
plt.show()
# %%

# Anova test of residential sale prices in 3 periods(Ex Ante, lockdown, Ex Post)
import statsmodels.api as sm
from statsmodels.formula.api import ols
model_1 = ols('price ~ C(period)', data=sales_trimmed).fit()
anova_table = sm.stats.anova_lm(model_1, typ=2)
anova_table
# %%
model_2 = ols('sale_num ~ C(period)', data=sales_num_trimmed).fit()
anova_table = sm.stats.anova_lm(model_2, typ=2)
anova_table
# %%
from statsmodels.formula.api import glm
fitmodel=glm(formula='price ~ C(period)', data=sales_trimmed)
fitmodel.summary()
fitmode2=glm(formula='sale_num ~ C(period)', data=sales_num_trimmed)
fitmode2.summary()
# %%
