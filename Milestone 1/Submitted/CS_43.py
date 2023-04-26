#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb


# # 
# # Functions

# **`PreprocessListCategories`** <br>
# `Function take df, list of categories columns and apply OHE on them, then return the df after modifications`

# In[ ]:


def PreprocessListCategories(df, lst):
    for col in lst:
        # Apply one-hot encoding to the "col" column in list and put the output in newdf
        newdf = df[col].str.get_dummies(sep=', ')
        df.drop(columns=[col], inplace=True)
        # Concatenate the one-hot encoded columns with the original DataFrame
        df = pd.concat([df, newdf], axis=1)
        
    return df


# **`CheckNullRows`**<br>
# `Function take df and print number of null rows`

# In[ ]:


def CheckNullRows(df):
    missing_rows = df.isnull().any(axis=1).sum()
    print('Number of rows that have null values: ', missing_rows)


# **`DropNullRows`**<br>
# `Function take df and drop all null rows`

# In[ ]:


def DropNullRows(df):
    df.dropna(inplace=True)


# **`ChangeDataType`**<br>
# `Function take df and change data types of some columns to appropriate data type`

# In[ ]:


def PreProcessAgrRating(df):
    # Age Rating 
    # Print current data type
    print('Data type of Age Rating is, ', df['Age Rating'].dtype)
    # Remove the + sign
    df['Age Rating'] = df['Age Rating'].str.replace('+', '', regex=False)
    # Convert Column datatype to int
    df['Age Rating'] = df['Age Rating'].astype(int)
    print('Data type of Age Rating after processing is ', df['Age Rating'].dtype)
    # Create a dictionary to map the age ratings to integers
    age_rating_map = {4: 1, 9: 2, 12: 3, 17: 4}
    # Replace each value with its category
    df['Age Rating'] = df['Age Rating'].replace(age_rating_map)
    # Print Age Rating
    print(df['Age Rating'].head())


# In[ ]:


def dropColumns(df, lst):
    for col in lst:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)


# In[ ]:


def ConvertToDateTime(df, lst):
    for col in lst:
        df[col] = pd.to_datetime(df[col].astype('datetime64'),dayfirst=True).astype('int64')


# In[ ]:


def GetColumnsNullsPerc(df):
    # print count of nulls for each column and percentage of them
    missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().mean())*100})
    return missing_data


# In[ ]:


def CheckListOfCategoriesColumn(df, col):
    print("Data type of ", col, "column is: ", df[col].dtype)
    all_languages = list(set(','.join(df['Languages'].fillna('').unique()).split(',')))
    print(col, "column has ", len(all_languages), "unique", col)


# In[ ]:


def PrintDfColumns(df):
    columns_list = df.columns.tolist()
    print(columns_list)


# In[ ]:


def FillColumnNulls(df, col):
    df[col] = df[col].fillna(0)


# In[ ]:


def DuplicatesDetectionAndRemoval(df):
    print("Number of duplicates rows: ", df.duplicated().sum())
    df.drop_duplicates(inplace = True, keep="first")


# In[ ]:


def outliers(dataset,col):
    fig, ax =plt.subplots(1,2)
    sns.boxplot( y=col, data=dataset,color="red", ax=ax[0])
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    dataset[col]=dataset[col].apply(lambda x: upper_bound if x> upper_bound else( lower_bound if x< lower_bound  else x))        
    sns.boxplot( y=col, data=dataset,color="blue", ax=ax[1])
    fig.show()


# In[ ]:


def avarage_Purchases(data,col):
    data[col]=data[col].fillna("0")
    data[col]=data[col].astype(str)
    data[col]=data[col].str.split(",")
    data[col]=[np.float64(x) for x in data[col]]
    data[col]=data[col].apply(lambda x: mean(x))
    data[col]=data[col].astype(float)
    return data


# In[ ]:


def reduceOHEColumns(df):
    X = df.iloc[:,14:] #columns of categories [languages, generes, ]
    zero_percentage = pd.DataFrame({'total_zeros': (X == 0).sum(), 'perc_zeros': (X == 0).mean() * 100})
    to_keep = zero_percentage['perc_zeros'] < 90
    
    updatedX = X.loc[:, to_keep]
    df = df.drop(df.columns[18:], axis=1)
    
    result = pd.concat([df, updatedX], axis=1)
    return result


# In[ ]:


def extract_country(links):
    us_words = []
    for link in links:
        parsed_url = urlparse(link)
        try:
            us_word = parsed_url.path.split('/')[1]
            us_words.append(us_word)
        except IndexError:
            print(f"Malformed link: {link}")
    return us_words


# In[ ]:


def extract_color(links):
    is_color_list = []
    for link in links:
        parsed_url = urlparse(link)
        netloc = parsed_url.netloc.lower()
        is_color_match = re.search(r"is\d", netloc)
        if is_color_match:
            is_color = is_color_match.group()
        else:
            is_color = None
        is_color_list.append(is_color)
    return is_color_list


# In[ ]:


def count_dev_games(df):
  # Create a dictionary to store the frequency of each developer
  developer_freq = df['Developer'].value_counts().to_dict()

  # Replace each developer name with its frequency in the dataset
  df['Developer'] = df['Developer'].map(developer_freq)
  return df


# In[ ]:


def frequent_words_in_name(df):
       # Download the stop words and tokenizer if needed
    nltk.download('stopwords')
    nltk.download('punkt')

    # Define the stop words list
    stop_words = set(stopwords.words('english'))

    # Create an empty Counter object to store the word frequencies
    word_freq = Counter()

    # Iterate over each row in the 'Name' column
    for name in df['Name']:
        # Check if the value is a string before tokenizing it
        if isinstance(name, str):
            # Tokenize the name string into words
            words = word_tokenize(name)

            # Filter out stop words and iterate over each word
            for word, pos in nltk.pos_tag(words):
                # Filter out stop words and check if the word is a noun or verb
                if word not in stop_words and (pos.startswith('N') or pos.startswith('V')):
                    # Add the word to the counter
                    word_freq[word.lower()] += 1

    # Get the 50 most frequent words
    most_common_words = [word[0] for word in word_freq.most_common(50)]

    # Replace each word in the 'Name' column with 1 if it matches one of the 50 most frequent words
    df['Name'] = df['Name'].apply(lambda x: sum(1 for word in word_tokenize(str(x).lower()) if word in most_common_words) + 1)

    return df


# In[ ]:


def scaling(df):
    #use Min-Max scaling, which scales the data to a range between 0 and 1.
    # Initialize the MinMaxScaler object
    scalerA = MinMaxScaler()
    scalerB = MinMaxScaler()
    # Fit and transform the features
    df['User Rating Count'] = scalerA.fit_transform(df[['User Rating Count']])
    df['Size'] = scalerB.fit_transform(df[['Size']])
    with open('sA.pkl', 'wb') as file:
        pickle.dump(scalerA,file)
    with open('sB.pkl', 'wb') as file:
        pickle.dump(scalerB,file)  
    return df


# In[ ]:


def apply_scaling(df):
    with open('sA.pkl', 'rb') as file:
        scaler_A = pickle.load(file)
    with open('sB.pkl', 'rb') as file:
        scaler_B = pickle.load(file)
    df['User Rating Count'] = scaler_A.transform(df[['User Rating Count']])
    df['Size'] = scaler_B.transform(df[['Size']])


# In[ ]:


def feature_transformation(df):
    # country name extraction
    df['URL'] = extract_country(df['URL'])
    df.rename(columns = {'URL':'Country'}, inplace = True)
    # color extraction
    df['Icon URL'] = extract_color(df['Icon URL'])
    df.rename(columns = {'Icon URL':'Color'}, inplace = True)
    # number of other games by dev
    df = count_dev_games(df)
    df.rename(columns = {'Developer':'Other by developer'}, inplace = True)
    df = scaling(df)
    return df


# In[ ]:


def extract_difficulty(description):
    difficulty = None
    if re.search(r"\b(hard|difficult|challenging|difficult|demanding|arduous|tough|grueling|strenuous|intense|brutal|hardcore|punishing)\b", description, re.IGNORECASE):
        difficulty = "Hard"
    elif re.search(r"\b(medium|moderate|average|intermediate|in-between|neither easy nor hard|fairly challenging|not too easy, not too hard|reasonably difficult|somewhat challenging|tolerable difficulty|manageable|adequate difficulty)\b", description, re.IGNORECASE):
        difficulty = "Medium"
    else:
        difficulty = "Easy"
    return difficulty


# In[ ]:


def selection(df1,df2):
    df = pd.concat([df1,df2] ,axis=1)
    corr = df.corr()
    top_features = corr.index[abs(corr['Average User Rating'])>0.1]
    top_features = top_features.delete(-1)
    return top_features


# 

# # 
# # Read Data

# In[ ]:


df = pd.read_excel('games-regression-dataset.xlsx')


# In[ ]:


df.shape


# # Split

# In[ ]:


# drop ID column because its unique value is equal to number of rows
df.drop(columns='ID', inplace=True)


# In[ ]:


df.shape
CheckNullRows(df)
DropNullRows(df)
CheckNullRows(df)
df.shape


# In[ ]:


DuplicatesDetectionAndRemoval(df)


# In[ ]:


Y = df['Average User Rating']
X = df.drop(columns='Average User Rating', inplace=False)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, shuffle=True, random_state=10)


# # 
# # Preprocessing Pipeline
# - **`1. Columns Analysis `**
# - **`2. Columns Nulls`**
# - **`3. Price Encoding`**
# - **`4. Outlires Detection & Removal`**
# - **`5. Deal With Categories`**
# - **`6. Feature Transformation`**

# ## 
# ### `1. Columns Analysis`

# In[ ]:


X_train.dtypes


# <br> 
# 
# **Age Rating**
# - Remove + sign 
# - Convert to int
# - Notice that the column has only 4 ages so we can categorize them

# In[ ]:


PreProcessAgrRating(X_train)


# <br>
# 
# **Languages**

# In[ ]:


CheckListOfCategoriesColumn(X_train, 'Languages')


# <br>
# 
# **Genres**

# In[ ]:


CheckListOfCategoriesColumn(X_train, 'Genres')


# <br>
# 
# **Primary Genre**

# In[ ]:


CheckListOfCategoriesColumn(X_train, 'Primary Genre')


# ## 
# **Dates**
# - Convert to date time data type

# In[ ]:


ConvertToDateTime(X_train, ['Original Release Date', 'Current Version Release Date'])


# ## 
# ### `2. Columns Nulls`

# In[ ]:


GetColumnsNullsPerc(X_train)


# In[ ]:


# because null percentage > 0.50
X_train.drop(columns=['Subtitle'], inplace=True)


# In[ ]:


GetColumnsNullsPerc(X_train)


# 
# <br> 
# 
# **In-app Purchases**
# - We can assum that any cell with null value, does not has any purshases. So replace all nulls with 0
# - Replce each cell with the mean

# In[ ]:


FillColumnNulls(X_train, 'In-app Purchases')
FillColumnNulls(X_train, 'Price')


# In[ ]:


GetColumnsNullsPerc(X_train)


# In[ ]:


X_train['Languages'] = X_train['Languages'].fillna('EN')


# In[ ]:


X_train = avarage_Purchases(X_train, 'In-app Purchases')


# In[ ]:


X_train['In-app Purchases'].dtype


# <br>
# 
# ## `3. Price Encoding`

# In[ ]:


X_train["Price"] = X_train["Price"].apply(lambda x: 1 if x > 0 else x)


# In[ ]:


X_train['Price'].value_counts()


# <br>
# 
# ## `4. Outlires Detection & Removal`

# In[ ]:


outliarlist=["User Rating Count","Size"]
for i in outliarlist:
    outliers(X_train,i)


# <br>
# 
# ## `5. Apply One Hot Encoding On Categories`

# In[ ]:


# replace tst with df
X_train = PreprocessListCategories(X_train,['Primary Genre', 'Genres', 'Languages'])
X_train = reduceOHEColumns(X_train)


# In[ ]:


X_train.shape


# <br>
# 
# ## `6. extracting usefull information from features`

# In[ ]:


X_train = feature_transformation(X_train)
print(X_train['Country'].unique)
#country column should be dropped because unique count is 0
dropColumns(X_train,['Country'])
X_train = frequent_words_in_name(X_train)
X_train.rename(columns = {'Name':'frequent words in Name'}, inplace = True)


# In[ ]:


X_train.head()


# ## Description Feature

# In[ ]:


# convert text to lowercase
X_train['Description'] = X_train['Description'].str.lower()
X_train['Game Difficulty'] = X_train['Description'].apply(extract_difficulty)
X_train.drop(columns='Description', inplace=True)
# Print the first 5 rows of the dataframe to verify the new columns have been added
X_train.head()


# In[ ]:


X_train['Game Difficulty'].value_counts()


# ## correlations

# In[ ]:


corrdf = pd.concat([X_train,Y_train] ,axis=1)


# In[ ]:


correlation_matrix = corrdf.corr()

# Get the correlation value between feature_1 and feature_2
correlation_value = correlation_matrix.loc['Size', 'Average User Rating']

print(f"The correlation between User Rating Count and Average User Rating is {correlation_value:.2f}")


# In[ ]:


correlation_matrix = corrdf.corr()

# Get the correlation value between feature_1 and feature_2
correlation_value = correlation_matrix.loc['Age Rating', 'Average User Rating']

print(f"The correlation between User Rating Count and Average User Rating is {correlation_value:.2f}")


# In[ ]:


correlation_matrix = corrdf.corr()

# Get the correlation value between feature_1 and feature_2
correlation_value = correlation_matrix.loc['In-app Purchases', 'Average User Rating']

print(f"The correlation between User Rating Count and Average User Rating is {correlation_value:.2f}")


# In[ ]:


correlation_matrix = corrdf.corr()

# Get the correlation value between feature_1 and feature_2
correlation_value = correlation_matrix.loc['Price', 'Average User Rating']

print(f"The correlation between User Rating Count and Average User Rating is {correlation_value:.2f}")


# ## Feature Selection

# In[ ]:


top_f = selection(X_train,Y_train)
print(top_f)
X_train = X_train[top_f]


# # Preprocess Test data

# In[ ]:


PreProcessAgrRating(X_test)
ConvertToDateTime(X_test, ['Original Release Date', 'Current Version Release Date'])
X_test.drop(columns=['Subtitle'], inplace=True)
FillColumnNulls(X_test, 'In-app Purchases')
FillColumnNulls(X_test, 'Price')
X_test['Languages'] = X_test['Languages'].fillna('EN')
X_test = avarage_Purchases(X_test, 'In-app Purchases')
X_test["Price"] = X_test["Price"].apply(lambda x: 1 if x > 0 else x)
X_test = PreprocessListCategories(X_test,['Primary Genre', 'Genres', 'Languages'])
X_test = reduceOHEColumns(X_test)
apply_scaling(X_test)


# # Train and test Models

# Linear Regression Model

# In[ ]:


ModelA = linear_model.LinearRegression()
ModelA.fit(X_train, Y_train)
p = ModelA.predict(X_train)
print("Model A accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelA = pickle.dumps(ModelA)


# In[ ]:


ModelA = pickle.loads(saved_modelA)
Y_pred = ModelA.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model A accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# SVR Model

# In[ ]:


from sklearn.svm import SVR
ModelB = SVR()
ModelB.fit(X_train, Y_train)
p = ModelB.predict(X_train)
print("Model B accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelB = pickle.dumps(ModelB)


# In[ ]:


ModelB = pickle.loads(saved_modelB)
Y_pred = ModelB.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model B accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# Decision Tree Model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
ModelC = DecisionTreeRegressor()
ModelC.fit(X_train, Y_train)
p = ModelC.predict(X_train)
print("Model C accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelC = pickle.dumps(ModelC)


# In[ ]:





# In[ ]:


ModelC = pickle.loads(saved_modelC)
Y_pred = ModelC.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model C accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
import pickle
# plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(ModelC, feature_names=['X1', 'X2'], filled=True)
plt.show()


# Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
ModelD = RandomForestRegressor()
ModelD.fit(X_train, Y_train)
p = ModelD.predict(X_train)
print("Model D accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelD = pickle.dumps(ModelD)


# In[ ]:


ModelD = pickle.loads(saved_modelD)
Y_pred = ModelD.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model D accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# K Neighbors Regressor Model

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
ModelE = KNeighborsRegressor()
ModelE.fit(X_train, Y_train)
p = ModelE.predict(X_train)
print("Model E accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelE = pickle.dumps(ModelE)


# In[ ]:


ModelE = pickle.loads(saved_modelE)
Y_pred = ModelE.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model E accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# XGBoost Regressor Model

# In[ ]:


ModelF = xgb.XGBRegressor(objective="reg:linear", random_state=42)
ModelF.fit(X_train, Y_train)
p = ModelF.predict(X_train)
print("Model F accurecy Test",r2_score(Y_train, p))
print('Mean Square Error Test', metrics.mean_squared_error(Y_train, p))
saved_modelF = pickle.dumps(ModelF)


# In[ ]:


ModelF = pickle.loads(saved_modelF)
Y_pred = ModelF.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model F accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# Lasso Regression Model

# In[ ]:


ModelG = Lasso(alpha = 10)
ModelG.fit(X_train, Y_train)
p = ModelG.predict(X_train)
print("Model G accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelG = pickle.dumps(ModelG)


# In[ ]:


a = X_train["User Rating Count"]


# In[ ]:


b = X_train["Original Release Date"]


# In[ ]:


import matplotlib.pyplot as plt

# check the dimensions of X_train
print(X_train.shape) # output: (n_samples, n_features)

# plot the data points
plt.scatter(a, Y_train, color='blue')


# In[ ]:


plt.scatter(b, Y_train, color='red')


# In[ ]:


# plot the regression line
plt.plot(a, ModelG.predict(X_train), color='green')


# In[ ]:



plt.plot(b, ModelG.predict(X_train), color='orange')


# In[ ]:



# set the axis labels and title
plt.xlabel('X features')
plt.ylabel('Y')
plt.title('Lasso Regression')

# show the plot
plt.show()


# In[ ]:


ModelG = pickle.loads(saved_modelG)
Y_pred = ModelG.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model G accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# Ridge Regression Model

# In[ ]:


ModelH = Ridge(alpha=10)
ModelH.fit(X_train, Y_train)
p = ModelH.predict(X_train)
print("Model H accurecy Train",r2_score(Y_train, p))
print('Mean Square Error Train', metrics.mean_squared_error(Y_train, p))
saved_modelH = pickle.dumps(ModelH)


# In[ ]:


ModelH = pickle.loads(saved_modelH)
Y_pred = ModelH.predict(X_test[top_f])
accuracy = r2_score(Y_test, Y_pred)
print("Model C accurecy Test",accuracy)
print('Mean Square Error Test', metrics.mean_squared_error(Y_test, Y_pred))


# In[ ]:




