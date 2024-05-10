import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import numpy as np
import os
def fill_missing(train):  
        train.drop_duplicates(keep=False, inplace=True)

        import pandas as pd
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=10)

        # Fill missing values using KNN imputation
        filled_data = pd.DataFrame(imputer.fit_transform(train.iloc[:,:-1]), columns=train.columns[:-1])

        train.iloc[:,:-1]=filled_data
        return train
def replace_outliers_with_max(df, multiple=4):
                        for column_name in df.columns:
                            # Calculate mean and standard deviation
                            mean_value = df[column_name].mean()
                            std_value = df[column_name].std()

                            # Calculate the threshold for outliers
                            threshold = mean_value + multiple * std_value

                            # Replace outliers with the maximum value
                            df.loc[df[column_name] > threshold, column_name] = mean_value + multiple * std_value

                        return df
def feauter_selection(train):
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=10)
        selector.fit_transform(train.iloc[:,:-1], train.iloc[:,-1])
        cols_idxs=np.arange(11)
        cols_idxs[-1]=-1
        cols_idxs[:10] = selector.get_support(indices=True)
        train=train.iloc[:,cols_idxs]
        train.drop(columns=train.columns[10:-1],inplace=True)   
        return train                     
def standerliztion(train):
        from sklearn.feature_selection import SelectKBest, f_classif
        min_max_scaler = StandardScaler()
        train.iloc[:,:-1] = min_max_scaler.fit_transform(train.iloc[:,:-1])
        return train
def balancing(train):
        from imblearn.over_sampling import SMOTE
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        y = train.iloc[:,-1]
        X = train.iloc[:,:-1]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y )  
        return X_resampled, y_resampled         
    
def clean_data(train):
    train=fill_missing(train)
    train.iloc[:,:-1] = replace_outliers_with_max(train.iloc[:,:-1])
    train=feauter_selection(train)
    train=standerliztion(train)
    x,y=balancing(train)

    return x,y