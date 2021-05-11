# coding:utf-8 
"""
author:Sam
date：2021/5/10
Target: use machine learning models to predict the underlying price movement direction.
Models: linear regression, lasso, ridge, logistic regression, SVM
"""

import numpy as np
import pandas as pd

# Techinical Indicators
import talib as ta

# Plotting graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# Data fetching
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

class Machine_learning_models():  # general machine learning models

    def __init__(self,x_train, y_train, x_test, y_test, features_df):
        """
            :param x_train: train set features
            :param y_train: train set labels
            :param x_test:  test set features
            :param y_test:  test set labels
            :param features_df: train set & test set features
            :return:
        """
        # input parameters
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.features_df = features_df

        # output paramaters
        self.intercept = None  # not set yet
        self.coeffient = None  # not set yet
        self.accuracy = None   # not set yet
        self.MAE = None        # not set yet
        self.MSE = None        # not set yet
        self.R_square = None   # not set yet


    def run(self):
        pass

class Linear_regression(Machine_learning_models):   # linear regression model, inherited from Machine_learing_models

    def run(self):
        linreg = LinearRegression()
        linreg.fit(self.x_train, self.y_train)
        self.intercept = linreg.intercept_
        self.coeffient = pd.DataFrame(list(zip(self.features_df, linreg.coef_)))
        print("The intercept of linear regression is:", self.intercept)
        print("The coeffient of linear regression is:\n", self.coeffient)

        y_pred = linreg.predict(self.x_test)
        self.accuracy = linreg.score(self.x_test, self.y_test)  # evaluate the results
        print("accuracy of linear regression is:", self.accuracy)

        self.MAE = metrics.mean_absolute_error(self.y_test, y_pred)
        self.MSE = metrics.mean_squared_error(self.y_test, y_pred)
        self.R_square = metrics.r2_score(self.y_test, y_pred)
        print("MAE of linear regression:", self.MAE)
        print('MSE of linear regression:', self.MSE)
        print('R-square of linear regression:', self.R_square)


class Lasso_regression(Machine_learning_models): # Lasso regression model, inherited from Machine_learing_models

    def run(self):
        lassocv = LassoCV()
        lassocv.fit(self.x_train, self.y_train)
        alpha = lassocv.alpha_
        print("The alpha for lasso is:",alpha)

        lasso = Lasso()
        lasso.fit(self.x_train, self.y_train)
        Lasso(alpha=alpha)
        self.intercept = lasso.intercept_
        self.coeffient = pd.DataFrame(list(zip(self.features_df, lasso.coef_)))
        print("The intercept of lasso regression is:", self.intercept)
        print("The coeffient of lasso regression is:\n", self.coeffient)

        y_pred = lasso.predict(self.x_test)
        self.accuracy = lasso.score(self.x_test, self.y_test)
        print("accuracy of lasso regression is:",self.accuracy)

        self.MAE = metrics.mean_absolute_error(y_test, y_pred)
        self.MSE = metrics.mean_squared_error(y_test, y_pred)
        self.R_square = metrics.r2_score(y_test, y_pred)
        print("MAE of lasso regression is:", self.MAE)
        print('MSE of lasso regression is:', self.MSE)
        print('R-square of lasso regression is:', self.R_square)
        print("------------------------------------------------")


class Ridge_regression(Machine_learning_models): # Ridge regression model, inherited from Machine_learing_models

    def run(self):
        ridgecv = RidgeCV()
        ridgecv.fit(self.x_train, self.y_train)
        alpha = ridgecv.alpha_
        print("The alpha for ridge is:",alpha)

        ridge = Ridge()
        ridge.fit(self.x_train, self.y_train)
        Ridge(alpha=alpha)
        self.intercept = ridge.intercept_
        self.coeffient = pd.DataFrame(list(zip(self.features_df, ridge.coef_)))
        print("The intercept of ridge regression is:", self.intercept)
        print("The coeffient of ridge regression is:\n", self.coeffient)

        y_pred = ridge.predict(self.x_test)
        self.accuracy = ridge.score(self.x_test,self.y_test)   #evaluate the results
        print("accuracy of lasso regression is:",self.accuracy)
        print("MAE of Ridge:", metrics.mean_absolute_error(self.y_test, y_pred))
        print('MSE of Ridge:', metrics.mean_squared_error(self.y_test, y_pred))
        print('R-square of Ridge:', metrics.r2_score(self.y_test, y_pred))
        print("------------------------------------------------")


class Logistic_regression(Machine_learning_models):  # Logistic regression model, inherited from Machine_learing_models

    def run(self):
        logreg = LogisticRegression()
        logreg.fit(self.x_train, self.y_train)
        self.intercept = logreg.intercept_
        self.coeffient = pd.DataFrame(list(zip(self.features_df.columns, np.transpose(logreg.coef_))))
        print("The intercept of logistic regression is:", self.intercept)
        print("The coeffient of logistic regression is:\n", self.coeffient)

        y_pred = logreg.predict(self.x_test)
        self.accuracy = logreg.score(self.x_test, self.y_test)  # evaluate the results

        print("accuracy of logistic regression:", self.accuracy)
        print('precision of logistic regression:', metrics.precision_score(self.y_test, y_pred, average='macro'))
        print('recall of logistic regression:', metrics.recall_score(self.y_test, y_pred, average='macro'))
        print('f1 score of logistic regression:', metrics.f1_score(self.y_test, y_pred, average='macro'))
        print('ROC_AUC of logistic regression:', metrics.roc_auc_score(self.y_test, y_pred, average='macro'))
        print("------------------------------------------------")

class SVM_linear(Machine_learning_models):   # SVM linear model, inherited from Machine_learing_models

    def run(self):
        svm_lin = svm.SVC(kernel='linear')
        svm_lin.fit(self.x_train, self.y_train)
        self.intercept = svm_lin.intercept_
        self.coeffient = pd.DataFrame(list(zip(self.features_df.columns, np.transpose(svm_lin.coef_))))
        print("The intercept of SVM is:", self.intercept)
        print("The coeffient of SVM is:\n", self.coeffient)

        y_pred = svm_lin.predict(self.x_test)
        print("accuracy of SVM:", metrics.accuracy_score(self.y_test, y_pred))
        print('precision of SVM:', metrics.precision_score(self.y_test, y_pred, average='macro'))
        print('recall of SVM:', metrics.recall_score(self.y_test, y_pred, average='macro'))
        print('f1 score of SVM:', metrics.f1_score(self.y_test, y_pred, average='macro'))
        print('ROC_AUC of SVM:', metrics.roc_auc_score(self.y_test, y_pred, average='macro'))
        print("------------------------------------------------")




if __name__ == '__main__':

    # basic setting
    underlying = 'AMD'
    start_time = '2000-01-01'
    end_time = '2021-01-01'
    train_set_proportion = 0.9


    # fetch date from yfinance
    stock_df = pdr.get_data_yahoo(underlying, start_time, end_time)
    stock_df = stock_df.dropna()
    del stock_df['Adj Close']
    print(stock_df)


    # select feature
    feature_names = ['Open-Close']
    stock_df['Open-Close'] = stock_df['Open'] - stock_df['Close'].shift(1)
    for n in [7, 15, 30]:
        # calculate moving average indicators
        stock_df['ma' + str(n)] = ta.SMA(stock_df['Close'].shift(1).values, timeperiod=n) / stock_df['Close']
        # calculate RSI
        stock_df['rsi' + str(n)] = ta.RSI(stock_df['Close'].shift(1).values, timeperiod=n)
        # add ma and RSI into feature_names list
        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
    print(feature_names)


    # feature and target dataframe
    stock_df = stock_df.dropna()
    features = stock_df[feature_names]
    targets = stock_df['Close']
    feature_target_df = stock_df[['Close']+feature_names]
    print(feature_target_df)


    # create training set
    split = int(train_set_proportion * int(len(features)))
    x_train, x_test, y_train, y_test = features[:split], features[split:], targets[:split], targets[split:]
    print(y_train)
    print(y_test)

    # linear regression
    linear = Linear_regression(x_train,y_train,x_test,y_test,features)
    linear.run()


    # lasso regression
    lasso = Lasso_regression(x_train,y_train,x_test,y_test,features)
    lasso.run()



    # ridge regression
    ridge = Ridge_regression(x_train, y_train, x_test, y_test, features)
    ridge.run()


    # logistic regression
    targets2 = np.where (stock_df['Close'].shift(-1) > stock_df['Close'],1,-1)  # target change to 1, -1 variables
    y_train2, y_test2 = targets2[:split], targets2[split:]
    logistic = Logistic_regression(x_train, y_train2, x_test, y_test2, features)
    logistic.run()

    # SVM regression
    SVM = SVM_linear(x_train, y_train2, x_test, y_test2, features)
    SVM.run()