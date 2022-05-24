from sklearn.model_selection import KFold
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import distance

def rmse(y,yhat):
    return(np.sqrt(((y - yhat)**2).mean()))

def rmsle(y,yhat):
    return np.sqrt(((np.log1p(yhat)-np.log1p(y))**2).mean())

def evaluate_model (n_splits, model, df_table, y):
    rmse_napake = []
    rmse_povprecje = []
    kf = KFold(n_splits, shuffle=True, random_state=0)
    for train, test in kf.split(df_table):
        X_train = df_table[train]
        X_test = df_table[test]
        y_train = y[train]
        y_test = y[test]
        y_povprecje = np.ones(y_test.size)*y_train.mean()
    
        reg = model.fit(X_train, y_train)
        y_prediction = reg.predict(X_test)
    
        rmse_napake.append(rmse(y_test, y_prediction))
        rmse_povprecje.append(rmse(y_test, y_povprecje))

        #koliko procentov napake od povprecnega modela je napaka nasega modela (lower the better)
    razmerje = 1/(np.mean(rmse_povprecje) / np.mean(rmse_napake))

    return np.mean(rmse_napake), razmerje

def bagging (nr_of_bags, percent, model, x_train, y_train, x_test):
    predictions = []
    (a,b) = x_train.shape
    for x in range(nr_of_bags):
        indexes = np.random.choice(a, size=int((percent/100)*a), replace=1)
        rez = model.fit(X_train[indexes], y_train[indexes])
        predictions.append(rez.predict(x_test))
    prediction = np.mean(np.array(predictions), axis=0)
    return prediction
        
def kNN (x_train, y_train, x_test, K, metrice):
    distance_matrix = distance.cdist(x_test, x_train, metrice)
    y_prediction = []
    sorted_indexes = np.argsort(distance_matrix, axis=1)
    for vrstica in sorted_indexes:
        y_prediction.append(np.mean(y_train[vrstica[:K]]))   
    return np.array(y_prediction)

def use_kNN (n_splits, K, data, y):
    
    rmse_napake = []
    rmse_povprecje = []
    
    kf = KFold(n_splits, shuffle=True, random_state=0)
    for train, test in kf.split(data):
        X_train = data[train]
        X_test = data[test]
        y_train = y[train]
        y_test = y[test]
        y_povprecje = np.ones(y_test.size)*y_train.mean()
    
        y_prediction = kNN(X_train, y_train, X_test, K, 'euclidean')
    
        rmse_napake.append(rmse(y_test, y_prediction))
        rmse_povprecje.append(rmse(y_test, y_povprecje))
    
    

    #koliko procentov napake od povprecnega modela je napaka nasega modela (lower the better)
    razmerje = 1/(np.mean(rmse_povprecje) / np.mean(rmse_napake))
    
    return np.mean(rmse_napake), razmerje


def use_bagging (n_splits, data, y, model, nr_of_bags, percent):
    rmse_napake = []
    rmse_povprecje = []
    kf = KFold(n_splits, shuffle=True, random_state=0)
    for train, test in kf.split(data):
        X_train = data[train]
        X_test = data[test]
        y_train = y[train]
        y_test = y[test]
        y_povprecje = np.ones(y_test.size)*y_train.mean()

        (a,b) = X_train.shape
        predictions = []
        for x in range(nr_of_bags):
            indexes = np.random.choice(a, size=int((percent/100)*a), replace=True)
            rez = model.fit(X_train[indexes], y_train[indexes])
            predictions.append(rez.predict(X_test))
        y_prediction = np.mean(np.array(predictions), axis=0)

        rmse_napake.append(rmse(y_test, y_prediction))
        rmse_povprecje.append(rmse(y_test, y_povprecje))


    #koliko procentov napake od povprecnega modela je napaka nasega modela (lower the better)
    razmerje = 1/(np.mean(rmse_povprecje) / np.mean(rmse_napake))

    return np.mean(rmse_napake), razmerje
    
    