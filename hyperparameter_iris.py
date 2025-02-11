#Importa a biblioteca pandas
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_validate
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sys

def get_database(db):
    return pd.read_csv(db)

def get_class(dataframe, classification):
    X = dataframe.drop(classification, axis=1)
    y = dataframe[classification]
    return X,y
    
def pre_process(dataframe, rem_cols, class_col, norm_cols):
    dataframe.drop(rem_cols, axis=1, inplace=True)
    
    lb = LabelEncoder()
    dataframe[class_col] = lb.fit_transform(dataframe[class_col])
    
    std = StandardScaler()
    dataframe[norm_cols] = std.fit_transform(dataframe[norm_cols])
    
def split_train(dataframe, classification):
    return train_test_split(dataframe, classification, test_size=0.3)

def generate_model(model, X, y):
    model = eval(model)
    model.fit(X,y)
    return model

def KFCross(model, X, y):
    kf = KFold(n_splits=10, shuffle=True)
    clf = cross_validate(
        eval(model),
        X,
        y,
        scoring='accuracy',
        cv = kf
    )
    
    return clf

def Skf(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    clf = cross_validate(
        eval(model),
        X,
        y,
        scoring='accuracy',
        cv = skf
    )
    
    return clf

def train_KFCross_model(X, y):
    cv = KFCross('DecisionTreeClassifier()',X,y)
    return np.mean(cv['test_score'])

def train_skf_model(X, y):
    cv = Skf('DecisionTreeClassifier()',X,y)
    return np.mean(cv['test_score'])

def train_dt_model(X_train,y_train,X_test,y_test):
    model = generate_model('DecisionTreeClassifier()',X_train,y_train)
    score = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    plt.figure()
    plot_tree(model,filled=True)
    plt.savefig('tree.png',format='png',bbox_inches = "tight")
    return score

def train_gridsearch_model(X,y,X_test,y_test):
    dt = DecisionTreeClassifier()
    param_grid = {'criterion':['gini','entropy','log_loss'],'splitter':['best','random'],'max_features':[sys.maxsize,1.0,'sqrt','log2',None]}
    g_search = GridSearchCV(estimator = dt, param_grid=param_grid, cv=10, refit=True)
    g_search.fit(X,y)
    model = g_search.best_estimator_
    return model.score(X_test,y_test)

def train_random_model(X,y,X_test,y_test):
    dt = DecisionTreeClassifier()
    param_grid = {'criterion':['gini','entropy','log_loss'],'splitter':['best','random'],'max_features':[sys.maxsize,1.0,'sqrt','log2',None]}
    r_search = RandomizedSearchCV(estimator=dt, param_distributions=param_grid, cv=10, refit=True)
    r_search.fit(X,y)
    model = r_search.best_estimator_
    return model.score(X_test, y_test)

def main():
    results = []
    df = get_database('Iris.csv')
    pre_process(df, ['Id'],'Species',['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    X,y = get_class(df,'Species')
    X_train,X_test, y_train,y_test = split_train(X,y)
    results.append(train_KFCross_model(X, y))
    results.append(train_skf_model(X,y))
    results.append(train_dt_model(X_train,y_train,X_test,y_test))
    results.append(train_gridsearch_model(X,y,X_test,y_test))
    results.append(train_random_model(X,y,X_test,y_test))
    
    print(results[np.argmax(results)],np.argmax(results))

if __name__ == "__main__":
    main()
    
