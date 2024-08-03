#!/usr/bin/env python
# coding: utf-8

# In[59]:


# import libraries

# 1. to handle the data
import pandas as pd
import numpy as np

# 2. To Viusalize the data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. import Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 5. Machine Learning
## sklearn.model_selection instead of model and it should be GridSearchCV instead of GridSearch and cross_val_score instead 
## of cross_val
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score

# 6. For Classification task.
from sklearn.linear_model import LogisticRegression #linear_model added
from sklearn.neighbors import KNeighborsClassifier #sklearn.neighbours
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
from xgboost import XGBClassifier

# 7. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[60]:


pip install xgboost


# In[61]:


pip install lightgbm


# In[62]:


url = "https://raw.githubusercontent.com/IEEECSMUJ/BreakingBug-ML/main/dataset.csv"
df = pd.read_csv(url)


# In[63]:


df.head()


# In[64]:


# print the first 5 rows of the dataframe
df.head()

# Exploring the data type of each column
df.info()

# Checking the data shape
df.shape

# Id column
df['id'].min(), df['id'].max()

# age column
df['age'].min(), df['age'].max()

# lets summerize the age column
df['age'].describe()


# In[65]:


import seaborn as sns

# Define custom colors
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]  # Example colors, you can adjust as needed


# In[66]:


sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color= 'Green')
plt.axvline(df['age'].mode()[0], color='Blue')


# In[67]:


# print the value of mean, median and mode of age column
print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode())


# In[68]:


# plot the histogram of age column using plotly and coloring this by sex

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()


# In[69]:


# Find the values of sex column
df['sex'].value_counts()


# In[70]:


# calculating the percentage fo male and female value counts in the data

male_count = 726
female_count = 194

total_count = male_count + female_count

# calculate percentages
male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

# display the results
print(f'Male percentage i the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')


# In[71]:


# Difference
difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')


# In[72]:


726/194


# In[73]:


# Find the values count of age column grouping by sex column
df.groupby('sex')['age'].value_counts()


# In[74]:


#dataset instead of dataseet, unique() instead of count()
df['dataset'].unique()


# In[75]:


# plot the countplot of dataset column
fig =px.bar(df, x='dataset', color='sex')
fig.show()


# In[76]:


# print the values of dataset column groupes by sex
print (df.groupby('sex')['dataset'].value_counts())


# In[77]:


# make a plot of age column using plotly and coloring by dataset

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()


# In[78]:


##Applied groupby() used value_counts() instead of pd.Series.mode and put line 1 in ()
print ("the mean median and mode of age column grouped by dataset column")
print("___________________________________________________________")
print ("Mean of the dataset: ",df.groupby('dataset')['age'].mean())
print("___________________________________________________________")
print ("Median of the dataset: ",df.groupby('dataset')['age'].median())
print("___________________________________________________________")
print ("Mode of the dataset: ",df.groupby('dataset')['age'].value_counts()) 
print("___________________________________________________________")


# In[79]:


# value count of cp column
df['cp'].value_counts()


# In[80]:


# count plot of cp column by sex column
sns.countplot(df, x='cp', hue= 'sex')


# In[81]:


# count plot of cp column by dataset column
sns.countplot(df,x='cp',hue='dataset')


# In[82]:


# Draw the plot of age column group by cp column

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()


# In[83]:


# lets summerize the trestbps column
df['trestbps'].describe()


# In[84]:


# Dealing with Missing values in trestbps column.
# find the percentage of misssing values in trestbps column
print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")


# In[85]:


# Impute the missing values of trestbps column using iterative imputer
# create an object of iteratvie imputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on trestbps column
imputer1.fit(df[['trestbps']])

# Transform the data
df['trestbps'] = imputer1.transform(df[['trestbps']])


# In[86]:


# Check the missing values in trestbps column
print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")


# In[87]:


# First lets see data types or category of columns
df.info()


# In[88]:


# let's see which columns has missing values
(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)


# In[89]:


# create an object of iterative imputer
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# fit transform on ca,oldpeak, thal,chol and thalch columns
## used fit_transform() instead of transfrom() and made it df[['column']] instead of column and imputer2 instead of imputer
df['ca'] = imputer2.fit_transform(df[['ca']])
df['oldpeak']= imputer2.fit_transform(df[['oldpeak']])
df['chol'] = imputer2.fit_transform(df[['chol']])
df['thalch'] = imputer2.fit_transform(df[['thalch']])


# In[90]:


# let's check again for missing values
(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")


# In[91]:


df['thal'].value_counts()


# In[92]:


df.head()


# In[93]:


# find missing values.
#df.null().sum()[df.null()()<0].values(ascending=true)



#missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()
##WE MADE A LIST OF COLUMNS OF MISSING VALUES
missing_data_cols=[ft for ft in df.columns if df[ft].isnull().sum() > 1]

missing_data_cols


# In[94]:


df['trestbps'].isnull().sum()


# In[95]:


# find categorical Columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols


# In[96]:


# find Numerical Columns
Num_cols = df.select_dtypes(exclude='object').columns.tolist()
Num_cols


# In[97]:


print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')


# In[109]:


# FInd columns
categorical_cols= ['thal', 'ca', 'slope', 'exang', 'restecg','thalch', 'chol', 'trestbps','fbs']
numerical_cols = ['oldpeak','age','restecg','fbs', 'cp', 'sex', 'num']


# In[110]:


df


# In[111]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

for col in categorical_cols:
    df[col] = imputer.fit_transform(df[[col]])


# In[246]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
# This function imputes missing values in categorical columnsdef impute_categorical_missing_data(passed_col):
##indentation fixed
passed_col = categorical_cols
def impute_categorical_missing_data(wrong_col):


    X = df(passed_col, axis=1)
    Y = df[passed_col]
    
    imputer = SimpleImputer(strategy='most_frequent')

    # Fit and transform the data
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
   

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()
    for col in X.columns:
           if X[col].dtype == 'object' :
               X[col] = onehotencoder.fit_transform(X[col].astype(str))

    if passed_col in bool_cols:
        Y = label_encoder.fit_transform(Y)
    
   

    imputer =  IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    for cols in other_missing_cols:
            cols_with_missing_values = Y[col].values.reshape(-1,1)
            imputed_values = imputer.fit_transform(cols_with_missing_values)
            X[col] = imputed_values[:, 0]
    else:
         pass

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    X = df_null.drop(passed_col, axis=1)

    for cols in Y.columns:
        if Y[col].dtype == 'object' :
            Y[col] = onehotencoder.fit_transform(Y[col].astype(str))

    for cols in other_missing_cols:
            cols_with_missing_value = Y[col].values.reshape(desired_shape)
            imputed_values = imputer.fit_transform(cols_with_missing_values)
            X[col] = imputed_values[:, 0]

    if len(df_null) < 0:
        df[passed] = classifier.predict(X)
        if passed in cols:
            df[passed] = df[passed].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]


# In[ ]:





# In[251]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
passed_col = categorical_cols
def impute_continuous_missing_data(wrong_col):
    
   
    
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestClassifier
    
    rf_classifer = RandomForestClassifier()
    
    


    

    X = df.drop(passed_col, axis=1)
    Y = df[passed_col]
    

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    for cols in X.columns:
        if X[cols].dtype == 'object' :
            X[cols] = ohe.fit_transform(X[[cols]])

    imputer =  IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    
   

  
    for cols in other_missing_cols:
        cols_with_missing_value = X[cols].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    

    print("MAE =", mae , "\n")
    print("RMSE =", rmse, "\n")
    print("R2 =", r2, "\n")

   

    for cols in other_missing_cols:
            cols_with_missing_values = X[cols].values.reshape(-1,1)
            imputed_values = imputer.fit_transform(cols_with_missing_values)
            X[cols] = imputed_values[:, 0]
    else:
         pass
    print(y_train)

    rf_classifer.fit(X_train, y_train)
    df[wrong_col] = rf_classifer.predict(X_test)
   

    

    return df[passed_col]


# In[ ]:





# In[252]:


df.info()


# In[253]:


df.isnull().sum().sort_values(ascending=False)


# In[254]:


# impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    if col in category.columns:
        category[col] = impute_categorical_missing_data(col)
    elif col in numerical_cols:
        df[col] = impute_continuous_missing_data(col)
    else:
        pass

df.isnull().sum().sort_values(ascending=False)


# In[241]:


sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})  # Change figure background color

palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

plt.figure(figsize=(10,8))

cols = df.columns

for i, col in enumerate(cols):
    plt.subplot(2,3,2) ##used 2 arguments earlier fixed it to 3
    sns.boxenplot(color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(i)

plt.show()
##E6E6FA


# In[255]:


# print the row from df where trestbps value is 0
df[df['trestbps']==0]


# In[256]:


# Remove the column because it is an outlier because trestbps cannot be zero.
df= df[df['trestbps']!=0]


# In[257]:


df.trestbps.describe()


# In[258]:


df.describe()


# In[259]:


# Set facecolors
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define the "night vision" color palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

# Use the "night vision" palette for the plots
plt.figure(figsize=(10, 8))
for i, col in enumerate(cols):
    plt.subplot(2,3,2)
    sns.boxenplot( color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()


# In[260]:


sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)

plt.figure(figsize=(10,8))



for i, col in enumerate(cols):
    plt.subplot(2,3,2)
    sns.boxenplot( color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.show()


# In[261]:


palette = ["#999999", "#666666", "#333333"]

sns.histplot(data=df,
             x='trestbps',
             kde=True,
             color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]


# In[262]:


# create a histplot trestbops column to analyse with sex column
sns.histplot(df, x='trestbps', kde=True, palette = "Spectral", hue ='sex')


# In[263]:


df.head()


# In[264]:


# split the data into X and y
X= df.drop('num', axis=1)
Y = df['num']


# In[272]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
"""encode X data using separate label encoder for all categorical columns and save it for inverse transform"""
# Task: Separate Encoder for all categorical and object columns and inverse transform at the end.
ohe = OneHotEncoder(sparse=False, drop='first')
Label_Encoder = LabelEncoder()
for cols in X.columns: #Y to X
    if X[cols].dtype == 'object' :
        reshaped_col = X[cols].values.reshape(-1, 1)
        X[cols] = ohe.fit_transform(reshaped_col.astype(str))
    else:
        pass


# In[274]:


# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[280]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import GradientBoostingClassifier


# In[281]:


#importing pipeline
from sklearn.pipeline import Pipeline

# import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error




import warnings
warnings.filterwarnings('ignore')


# In[284]:


# create a list of models to evaluate

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]




best_model = None
best_accuracy = 0.0


# In[286]:


# Iterate over the models and evaluate their performance
for name, model in models:
    # Create a pipeline for each model
    pipeline = Pipeline([
        # ('imputer', SimpleImputer(strategy='most_frequent')),  # Uncomment if needed
        # ('encoder', OneHotEncoder(handle_unknown='ignore')),  # Uncomment if needed
        ('model', model)
    ])
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    # Calculate mean accuracy from cross-validation scores
    mean_accuracy = scores.mean()
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Print the performance metrics
    print("Model:", name)
    print("Cross-Validation Accuracy: {:.2f}".format(mean_accuracy))
    print("Test Accuracy: {:.2f}".format(accuracy))
    print()







# In[288]:


#Check if the current model has the best accuracy
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model = pipeline


# In[289]:


# Retrieve the best model
print("Best Model: ", best_model)


# In[290]:


categorical_columns = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']


# In[300]:


def evaluate_classification_models(X, y, categorical_columns):
    # Encode categorical columns
    X_encoded = X.copy()
    onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    for col in categorical_columns:
        X_encoded[col] = onehotencoder.fit_transform(X[[col]].astype(str))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }


    # Train and evaluate models
    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model


# In[301]:


# Example usage:
results, best_model = evaluate_classification_models(X, Y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)


# In[302]:


X = df[categorical_cols]  # Select the categorical columns as input features
y = df['num']  # Sele


# In[307]:


def hyperparameter_tuning(X, y, categorical_columns, models):
    # Define dictionary to store results
    results = {}

    # Encode categorical columns
    X_encoded = X.copy()
    onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    for col in categorical_columns:
        X_encoded = X_encoded.join(
            pd.DataFrame(onehotencoder.fit_transform(X[[col]].astype(str)), 
                         columns=onehotencoder.get_feature_names_out([col]), 
                         index=X.index)
        )
        X_encoded.drop(columns=[col], inplace=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Perform hyperparameter tuning for each model
    for model_name, model in models.items():
        # Define parameter grid for hyperparameter tuning
        param_grid = {}
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9]}
        elif model_name == 'NB':
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'XGBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'GradientBoosting':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'AdaBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}

        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get best hyperparameters and evaluate on test set
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store results in dictionary
        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results


# In[309]:


# Define models dictionary
models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }


# In[ ]:


# Example usage:
results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()


# In[ ]:




