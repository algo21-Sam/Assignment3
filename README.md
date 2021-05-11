# Assignment3
## 1.Abstract
This project is intented to **utilize machine learning to predict stock price movement direction** by 5 models, namely **linear regression, lasso, ridge, logistic regression, SVM**.

## 2.Reference
Machine Learning For Stock Price Prediction Using Regression

https://blog.quantinsti.com/machine-learning-trading-predict-stock-prices-regression/


## 3.Setting
underlying: AMD

historical data time period: 2000-01-01 to 2020-12-31

test set proportion: 90%


## 4.Code logic and explaination
I define a general class called **Machine_learning_models**, which takes the input of train set and test set data, as every machine learn model does. Then, **five models inherit this class and have their own run function**. The most useful tool is sklearn package, which includes all the models for me to fit the dataset.

After defining such classes, I first set the underlying, time period and train data proportion. Then, fetch the market data I need for yahoo finance API and construct several features with the help of talib package. **The target for linear regression, lasso and ridge is stock close price, while target for logistic regression, SVM is the direction of stock price movement**. 


## 5.Result
### 5.1 Linear regression

The intercept of linear regression is: -2.623034338936815

The coeffient of linear regression is:

             0         1
             
0  Open-Close  0.251731

1         ma7  8.258592

2        rsi7  0.288443

3        ma15 -2.075800

4       rsi15 -1.120457

5        ma30 -7.529708

6       rsi30  1.142333

accuracy of linear regression is: -2.1970429601741657

MAE of linear regression: 33.470553697048025

MSE of linear regression: 1578.8154124250482

R-square of linear regression: -2.1970429601741657


### 5.2 lasso

The alpha for lasso is: 0.08301710804244218

The intercept of lasso regression is: -2.4182101282626967

The coeffient of lasso regression is:

             0         1
             
0  Open-Close  0.000000

1         ma7 -0.000000

2        rsi7  0.014478

3        ma15 -0.000000

4       rsi15 -0.315332

5        ma30 -0.000000

6       rsi30  0.578410

accuracy of lasso regression is: -2.257991733337312

MAE of lasso regression is: 33.77553352320676

MSE of lasso regression is: 1608.9141203989727

R-square of lasso regression is: -2.257991733337312


### 5.3 ridge

The alpha for ridge is: 1.0

The intercept of ridge regression is: -2.3002802342381603

The coeffient of ridge regression is:

             0         1
             
0  Open-Close  0.217570

1         ma7  6.475308

2        rsi7  0.280090

3        ma15 -1.476884

4       rsi15 -1.098716

5        ma30 -6.868932

6       rsi30  1.132809

accuracy of lasso regression is: -2.1977944987533027

MAE of Ridge: 33.473687920778865

MSE of Ridge: 1579.1865493495604

R-square of Ridge: -2.1977944987533027


### 5.4 logistic regression

The intercept of logistic regression is: [-0.30732811]

The coeffient of logistic regression is:

             0                         1
             
0  Open-Close     [0.12114399526654156]

1         ma7    [-0.20751144086660855]

2        rsi7   [-0.003824866732637772]

3        ma15    [-0.14694346085607926]

4       rsi15  [0.00042326953529272284]

5        ma30      [0.2602527494014587]

6       rsi30     [0.01089805472989139]

accuracy of logistic regression: 0.47718631178707227

precision of logistic regression: 0.4709008830392411

recall of logistic regression: 0.47236704900938475

f1 score of logistic regression: 0.467483939845749

ROC_AUC of logistic regression: 0.4723670490093847


### 5.5 SVM
The intercept of SVM is: [-2.33341188]

The coeffient of SVM is:

             0                        1
             
0  Open-Close     [0.3417058768196455]

1         ma7   [-0.34211311538138034]

2        rsi7  [-0.004754422952828463]

3        ma15    [-0.5970504354914965]

4       rsi15   [-0.04720405058469623]

5        ma30     [0.5201103880900746]

6       rsi30    [0.10355106922361301]

accuracy of SVM: 0.4714828897338403

precision of SVM: 0.46174993643529116

recall of SVM: 0.46514019232997333

f1 score of SVM: 0.45554330990110803

ROC_AUC of SVM: 0.46514019232997333



## 6.Tips
Sometimes the implementation would come across some network problem, for some reasons I don't quite understand. Just try several times!
