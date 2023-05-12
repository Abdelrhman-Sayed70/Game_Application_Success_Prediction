from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd
from statistics import mean, mode
import re
from sklearn.feature_selection import f_regression, f_classif, SelectKBest

scaler = None
dev_dict = None
cols = None
ver_m = None
org_m = None
dev_m = None
age_m = None


def base(df):
    drop_cols(df)
    fill_nulls(df)
    df = avarage_Purchases(df, 'In-app Purchases')
    ConvertToDateTime(df, ['Original Release Date', 'Current Version Release Date'])
    PreProcessAgrRating(df)
    df['Description'] = df['Description'].str.lower()
    df['Game Difficulty'] = df['Description'].apply(extract_difficulty)
    df.drop(columns='Description', inplace=True)
    return df

def PrintDfColumns(df):
    columns_list = df.columns.tolist()
    print(columns_list)


def extract_difficulty(description):
    difficulty = None
    if re.search(r"\b(hard|difficult|challenging|demanding|arduous|tough|grueling|strenuous|intense|brutal|hardcore|punishing)\b",
                 description, re.IGNORECASE):
        difficulty = 2
    elif re.search(r"\b(medium|moderate|average|intermediate|in-between|neither easy nor hard|fairly challenging|not too easy, not too hard|reasonably difficult|somewhat challenging|tolerable difficulty|manageable|adequate difficulty)\b",
                   description, re.IGNORECASE):
        difficulty = 1
    else:
        difficulty = 0
    return difficulty


def DuplicatesDetectionAndRemoval(df):
    df.drop_duplicates(inplace=True, keep="first")


def DropNullRows(df):
    df.dropna(inplace=True)


def on_train(df):
    df = PreprocessListCategories_train(df, ['Genres', 'Languages'])
    df = scaling(df)
    df = count_dev_games(df)
    df = outliers(df, "User Rating Count")
    df = outliers(df, "Size")
    vermean = pd.to_datetime(df['Current Version Release Date']).mean()
    df['Current Version Release Date'] = df['Current Version Release Date'].fillna(vermean)
    global ver_m
    ver_m = pickle.dumps(vermean)
    orgmean = pd.to_datetime(df['Original Release Date']).mean()
    df['Original Release Date'] = df['Original Release Date'].fillna(orgmean)
    global org_m
    org_m = pickle.dumps(orgmean)
    devmode = df['Developer'].mode()
    df['Developer'] = df['Developer'].fillna(devmode)
    global dev_m
    dev_m = pickle.dumps(devmode)
    agemode = df['Age Rating'].mode()
    df['Age Rating'] = df['Age Rating'].fillna(agemode)
    global age_m
    age_m = pickle.dumps(agemode)
    return df


def on_test(df):
    df = apply_scaling(df)
    df = replace_dev(df)
    df = PreprocessListCategories_test(df, ['Genres', 'Languages'])
    vermean = pickle.loads(ver_m)
    df['Current Version Release Date'] = df['Current Version Release Date'].fillna(vermean)
    orgmean = pickle.loads(org_m)
    df['Original Release Date'] = df['Original Release Date'].fillna(orgmean)
    devmode = pickle.loads(dev_m)
    df['Developer'] = df['Developer'].fillna(devmode)
    agemode = pickle.loads(age_m)
    df['Age Rating'] = df['Age Rating'].fillna(agemode)
    return df


def drop_cols(df):
    df.drop(columns=['ID', 'URL', 'Name', 'Icon URL', 'Subtitle', 'Primary Genre'], inplace=True)


def scaling(df):
    s = MinMaxScaler()
    # Fit and transform the features
    df[['User Rating Count', 'In-app Purchases', 'Size', 'Original Release Date',
                        'Current Version Release Date']] = s.fit_transform(df[['User Rating Count', 'In-app Purchases', 'Size', 'Original Release Date',
                        'Current Version Release Date']])
    global scaler
    scaler = pickle.dumps(s)
    return df


def apply_scaling(df):
    s = pickle.loads(scaler)
    df[['User Rating Count', 'In-app Purchases', 'Size', 'Original Release Date',
        'Current Version Release Date']] = s.transform(df[['User Rating Count', 'In-app Purchases', 'Size', 'Original Release Date',
                        'Current Version Release Date']])
    return df


def fill_nulls(df):
    df['Languages'] = df['Languages'].fillna('EN')
    df['Price'] = df['Price'].fillna(0)
    df['In-app Purchases'] = df['In-app Purchases'].fillna(0)
    df['User Rating Count'] = df['User Rating Count'].fillna(0)
    df['Size'] = df['Size'].fillna(0)
    df['Genres'] = df['Genres'].fillna('Games')

def outliers(dataset, col):
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset[col] = dataset[col].apply(
        lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return dataset


def avarage_Purchases(data, col):
    data[col] = data[col].astype(str)
    data[col] = data[col].str.split(",")
    data[col] = [np.float64(x) for x in data[col]]
    data[col] = data[col].apply(lambda x: mean(x))
    data[col] = data[col].astype(float)
    return data


def ConvertToDateTime(df, lst):
    for col in lst:
        df[col] = pd.to_datetime(df[col].astype('datetime64')).astype('int64')


def PreprocessListCategories_train(df, lst):
    for col in lst:
        # Apply one-hot encoding to the "col" column in list and put the output in newdf
        newdf = df[col].str.get_dummies(sep=', ')
        df.drop(columns=[col], inplace=True)
        # Concatenate the one-hot encoded columns with the original DataFrame
        df = pd.concat([df, newdf], axis=1)
    global cols
    cols = pickle.dumps(df.columns.tolist())
    return df


def PreprocessListCategories_test(df, lst):
    for col in lst:
        # Apply one-hot encoding to the "col" column in list and put the output in newdf
        newdf = df[col].str.get_dummies(sep=', ')
        df.drop(columns=[col], inplace=True)
        # Concatenate the one-hot encoded columns with the original DataFrame
        df = pd.concat([df, newdf], axis=1)
    df = only_showed_cols(df)
    return df


def only_showed_cols(df):
    x = pickle.loads(cols)
    missing_cols = set(x) - set(df.columns)
    # adding columns seen at train but not in test
    for c in missing_cols:
        df[c] = 0
    # removing columns seen at test but not in train
    return df[x]


def PreProcessAgrRating(df):
    # Remove the + sign
    df['Age Rating'] = df['Age Rating'].str.replace('+', '', regex=False)
    # Convert Column datatype to int
    df['Age Rating'] = df['Age Rating'].astype(int)
    # Create a dictionary to map the age ratings to integers
    age_rating_map = {4: 1, 9: 2, 12: 3, 17: 4}
    # Replace each value with its category
    df['Age Rating'] = df['Age Rating'].replace(age_rating_map)
    return df


def count_dev_games(df):
    # Create a dictionary to store the frequency of each developer
    developer_freq = df['Developer'].value_counts().to_dict()
    df['Developer'] = df['Developer'].map(developer_freq)
    global dev_dict
    dev_dict = pickle.dumps(developer_freq)
    return df


def replace_dev(df):
    # Replace each developer name with its frequency in the dataset
    developer_freq = pickle.loads(dev_dict)
    df['Developer'] = df['Developer'].map(developer_freq)
    df['Developer'] = df['Developer'].fillna(0)
    return df


def feature_selection_regression(df1,Y1):
  k = 100
  selector = SelectKBest(f_regression, k=k)
  selector.fit(df1, Y1)
  X_train_kbest = selector.transform(df1)
  column_names = df1.columns
  top_feature_indices = selector.get_support(indices=True)
  top_feature_names = column_names[top_feature_indices]
  return top_feature_names


def feature_selection_classification(df1,Y1):
  k = 63
  selector = SelectKBest(f_classif, k=k)
  selector.fit(df1, Y1)
  X_train_kbest = selector.transform(df1)
  column_names = df1.columns
  top_feature_indices = selector.get_support(indices=True)
  top_feature_names = column_names[top_feature_indices]
  return top_feature_names
