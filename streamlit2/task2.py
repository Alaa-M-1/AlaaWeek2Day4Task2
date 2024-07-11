import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import pandas as pd
import os
print(os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv(r'C:\Users\aaii1\Desktop\streamlit2\insurance.csv')



# Display the first few rows and check the columns
print(data.head())
print(data.columns)

st.title('Week2 - Day4 - Task2')

st.write("""
Explore different classifier and datasets
Which one is the best?
""")




classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Linear Regression',
    'Random Forest Regression',
    'Gradient Boosting Regression')
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Separate features and target variable
X = data.drop('charges', axis=1)
y = data['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Step 3: Building Regression Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=42)
}


regressors = {}
for model_name, model in models.items():
    regressors[model_name] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

# Lists to store metrics
model_names = []

rmse_scores = []
r2_scores = []
mean_values = []


for model_name, regressor in regressors.items():
    print(f"Training {model_name}: ")
    regressor.fit(X_train, y_train)
    print("Model trained.")

    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mean_pred = np.mean(y_pred)

    # Store metrics
    model_names.append(model_name)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    mean_values.append(mean_pred)

    #print(f"{model_name}    " +  classifier_name)
    if(f"{model_name}"==classifier_name):
        st.write(f"Metrics for {model_name}:")
        st.write(f"  Mean: {mse}")
        st.write(f"  RMSE: {rmse}")
        st.write(f"  R^2 Score: {r2}")
    

    if model_name == 'Random Forest Regression':


        # Extract feature names after one-hot encoding
        categorical_encoder = regressor.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        feature_names = categorical_features
        feature_names = numeric_features + feature_names

        feature_importances = regressor.named_steps['regressor'].feature_importances_
        print("Random Forest Feature Importances:")
        for i, feature_name in enumerate(feature_names):
            print(f"{feature_name}: {feature_importances[i]}")
        print()
        






def get_classifier(clf_name):
    clf = None
    if clf_name == 'SVM':
        pass
        
    elif clf_name == 'KNN':
        pass
    else:
        pass
        
    

#clf = get_classifier(classifier_name)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)



#### PLOT DATASET ####
from sklearn.preprocessing import MinMaxScaler

# Convert metrics to DataFrame for easier plotting
metrics_df = pd.DataFrame({
    'Model': model_names,
    'RMSE': rmse_scores,
    'R2 Score': r2_scores,
    'Mean': mean_values
})

# Normalize the metrics
scaler = MinMaxScaler()
metrics_df[['RMSE', 'R2 Score', 'Mean']] = scaler.fit_transform(metrics_df[['RMSE', 'R2 Score', 'Mean']])

# Plotting RMSE, R^2 scores, and Mean

# Plotting normalized RMSE, R^2 scores, and Mean
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for normalized metrics
metrics_df.set_index('Model').plot(kind='bar', ax=ax)

# Add title and labels
plt.title('Normalized Model Performance Comparison')


plt.tight_layout()
plt.show()
#plt.show()
st.write(f"## Insurance Dataset")
st.pyplot(fig)