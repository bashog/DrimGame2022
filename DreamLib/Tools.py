##### LIBRAIRIES #####

# Models
from pickle import FALSE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense,InputLayer

# Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

# Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# Grid Search
from sklearn.model_selection import GridSearchCV

# Data processing
import pandas as pd
import numpy as np

# Normalize
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Plot
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
sns.set_theme()
plt.rcParams['font.family'] = 'serif'

##### PROCESSING FUNCTIONS #####

def clean_data(data:pd.core.frame.DataFrame,start:int,chronique,col_used,split=0.2,norm="MinMax"):
  """
  - Returns the data normalized and prepared for t -> X_train,X_test,y_train,y_test
  - X_train and X_test are DataFrame and y_train,y_test are Series
  - Chronique can take these values (it's an object):
    * b'CHR2'
    * b'CHR8'
    * b'Total'
  - You have to enter the initial
  - Norm can take these values:
    * MinMax for MinMaxScaler
    * StdSca
    * Not if you don't want to normalize your data
  """
  
  # Data processing
  X = data[data.CHRONIQUE == chronique]
  X = X.iloc[start:,]
  X = X.set_index(X.TRIMESTRE)
  Y = X.DR
  X = X[col_used]
  
  # Train Split
  X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=split,shuffle=False)
  index_train = X_train.index
  index_test = X_test.index

  # Normalization
  if norm == "Not":
    return X_train, X_test, y_train, y_test
  
  elif norm == 'MinMax':
    scaler_train = MinMaxScaler()
    scaler_test = MinMaxScaler()
    X_train = pd.DataFrame(scaler_train.fit_transform(X_train),index=index_train,columns=col_used)
    X_test = pd.DataFrame(scaler_test.fit_transform(X_test),index=index_test,columns=col_used)

  elif norm == 'StdSca':
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    X_train = pd.DataFrame(scaler_train.fit_transform(X_train),index=index_train,columns=col_used)
    X_test = pd.DataFrame(scaler_test.fit_transform(X_test),index=index_test,columns=col_used)

  else:
    print("This norm doesn't exist")

  return X_train, X_test, y_train, y_test
  
##### REGRESSION FUNCTIONS #####

def summary_ml(X_train,y_train,X_test,y_test,models=["lin","rid","xgb","svr","knn","cat",'tre'],parameters={},metrics=["rmse","mse","mae","r2"]):
    """
    - Returns a DataFrame with all the model you want to train, and his score
    - Models is a list with names of models you want to train. Values in models can be:
      * lin for LinearRegression
      * rid for Ridge
      * las for Lasso
      * ela for ElasticNet
      * xgb for XGBRegressor
      * svr for LinearSVR
      * knn for KNeighborsRegressor
      * cat for CatBoostRegression
      * tre for DecisionTreeRegressor
    - Parameters is a dictionnary containing several dictionnaries for models you want to use. Values of each dictionnary containing parameters can be:
      * p_lin for LinearRegression
      * p_rid for Ridge
      * p_las for Lasso
      * p_ela for ElasticNet
      * p_xgb for XGBRegressor
      * p_svr for LinearSVR
      * p_knn for KNeighborsRegressor
      * p_cat for CatBoostRegression
      * p_tre for DecisionTreeRegressor
    - Metrics is a list to choose the right metric to use. metric is equal to ["mse"] by default. Currently two metrics are taken into account:
      * mse for mean_squared_error
      * mae for mean_absolute_error
      * rmse for root_mean_squarred_error
      * r2 for r2_score

    !BE CAREFUL! Lasso and ElasticNet returns systematically a constant prediction so we do not include them in the predictions
    """
  
 	# Index used to create the DataFrame
    Index = []
    Columns = {metric: [] for metric in metrics}

    for model in models:
        unknown = 0
    # Training the model

        if model == "lin":
            Index.append("LinearRegression")
            if "p_lin" in parameters:
                M = LinearRegression(**parameters["p_lin"])
                M = M.fit(X_train,y_train)
            else:
                M = LinearRegression()
                M = M.fit(X_train,y_train)

        elif model == "rid":
            Index.append("Ridge")
            if "p_rid" in parameters:
                M = Ridge(**parameters["p_rid"])
                M = M.fit(X_train,y_train)
            else:
                M = Ridge()
                M = M.fit(X_train,y_train)

        elif model == "las":
            Index.append("Lasso")
            if "p_las" in parameters:
                M = Lasso(**parameters["p_las"])
                M = M.fit(X_train,y_train)
            else:
                M = Lasso()
                M = M.fit(X_train,y_train)

        elif model == "ela":
            Index.append("ElasticNet")
            if "p_ela" in parameters:
                M = ElasticNet(**parameters["p_ela"])
                M = M.fit(X_train,y_train)
            else:
                M = ElasticNet()
                M = M.fit(X_train,y_train)

        elif model == "xgb":
            Index.append("XGBRegressor")
            if "p_xgb" in parameters:
                M = XGBRegressor(**parameters["p_xgb"])
                M = M.fit(X_train,y_train)                
            else:
                M = XGBRegressor()
                M = M.fit(X_train,y_train)

        elif model == "svr":
            Index.append("LinearSVR")
            if "p_svr" in parameters:
                M = LinearSVR(**parameters["p_svr"])
                M = M.fit(X_train,y_train)                
            else:
                M = LinearSVR()
                M = M.fit(X_train,y_train) 

        elif model == "knn":
            Index.append("KNeighborsRegressor")
            if "p_knn" in parameters:
                M = KNeighborsRegressor(**parameters["p_knn"])
                M = M.fit(X_train,y_train)          
            else:
                M = KNeighborsRegressor()
                M = M.fit(X_train,y_train)

        elif model == "cat":
            Index.append("CatBoostRegressor")
            if "p_cat" in parameters:
                M = CatBoostRegressor(**parameters["p_cat"])
                M = M.fit(X_train,y_train)                
            else:
                M = CatBoostRegressor(verbose=0)
                M = M.fit(X_train,y_train)
                

        elif model == "tre":
            Index.append("DecisionTreeRegressor")
            if "p_tre" in parameters:
                M = DecisionTreeRegressor(**parameters["p_tre"],random_state=False)
                M = M.fit(X_train,y_train)                
            else:
                M = DecisionTreeRegressor(random_state=False)
                M = M.fit(X_train,y_train)

        else:
            print("The model is unknown")
            unknown = 1
            
        if unknown == 0:
          y_pred = M.predict(X_test)
        
        else:
          y_pred != (y_test == False)

        # Calculate the score for each metric
        for metric in metrics:
            if metric == "rmse":
                Columns["rmse"].append(mean_squared_error(y_test,y_pred,squared=False)) 
            
            elif metric == "mse":
                Columns["mse"].append(mean_squared_error(y_test,y_pred))

            elif metric == "mae":
                Columns["mae"].append(mean_absolute_error(y_test,y_pred)) 

            elif metric == "r2":
                Columns["r2"].append(r2_score(y_test,y_pred)) 

            else:
                Columns[metric].append(None)

    # We transform the dictionnary into a DataFrame
    df = pd.DataFrame.from_dict(Columns)
    df["Index"] = Index
    df = df.set_index(df.Index)
    df = df.drop(["Index"],axis = 1)
    # Condition on r2 because we want the biggest r2
    if df.columns[0] == "r2":
      df = df.sort_values(by=df.columns[0],ascending=False)
    else:
      df = df.sort_values(by=df.columns[0],ascending=True)
    
    return df

def params_grid(X_train,y_train,X_test,y_test,model,parameters,cv=2,metric="rmse"):
  """
  - Returns the optimal parameters and the score of the model
  - Model is a string used to define the model we want to use. Values in models can be:
    * lin for LinearRegression
    * rid for Ridge
    * las for Lasso
    * ela for ElasticNet
    * xgb for XGBRegressor
    * svr for LinearSVR
    * knn for KNeighborsRegressor
    * cat for CatBoostRegressor
    * tre for DecisionTreeRegressor
  - Metric can take these values:
    * mse for neg_mean_squared_error
    * mae for neg_median_absolute_error
    * rmse for neg_root_mean_squared_error
    * r2 for r2_score
  """

  metrics = {"mse":"neg_mean_squared_error","mae":"neg_median_absolute_error","rmse":"neg_root_mean_squared_error","r2":"r2_score"}

  if model == "lin":
        M = LinearRegression()
        Grid = GridSearchCV(M, parameters,verbose = 1,scoring = metrics[metric],cv=cv)
        Gridfit = Grid.fit(X_train,y_train)

  elif model == "rid":
        M = Ridge()
        Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)
        Gridfit = Grid.fit(X_train,y_train)

  elif model == "las":
        M = Lasso()
        Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv) 
        Gridfit = Grid.fit(X_train,y_train)

  elif model == "ela":
        M = ElasticNet()
        Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)              
        Gridfit = Grid.fit(X_train,y_train)

  elif model == "xgb":
        M = XGBRegressor()
        Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)
        Gridfit = Grid.fit(X_train,y_train)

  elif model == "svr": 
        M = LinearSVR()
        Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)
        Gridfit = Grid.fit(X_train,y_train)

  elif model == "knn": 
        M = KNeighborsRegressor()
        Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)
        Gridfit = Grid.fit(X_train,y_train)
  
  elif model == "cat": 
      M = CatBoostRegressor()
      Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)
      Gridfit = Grid.fit(X_train,y_train)

  elif model == "tre": 
      M = DecisionTreeRegressor(random_state=False)
      Grid = GridSearchCV(M, parameters, verbose = 1,scoring = metrics[metric],cv=cv)
      Gridfit = Grid.fit(X_train,y_train)

  else:
      print("The model is unkwown")
    
  y_pred = Gridfit.predict(X_test)

  if metric == "rmse":
      score = mean_squared_error(y_test,y_pred,squared=False)

  elif metric == "mse":
      score = mean_squared_error(y_test,y_pred)

  elif metric == "mae":
      score = mean_absolute_error(y_test,y_pred)

  elif metric == "r2":
      score = r2_score(y_test,y_pred)

  else:
      score = None
  
  return Gridfit.best_params_,score

def y_pred(X_train,y_train,X_test,y_test,model,parameters={}):
    """
    - Returns a prediction according to the model
    - Models is a list with names of models you want to train. Values in models can be:
      * lin or LinearRegression for LinearRegression
      * rid or Ridge for Ridge
      * las or Lasso for Lasso
      * ela or ElasticNet for ElasticNet
      * xgb or XGBRegressor for XGBRegressor
      * svr or LinearSVR for LinearSVR
      * knn or KNeighborsRegressor for KNeighborsRegressor
      * cat or CatBoostRegression for CatBoostRegression
      * tre or DecisionTreeRegressor for DecisionTreeRegressor
    - Parameters is a dictionnary containing the parameters you want for the model.
    """

    # Training the model
    unknown = 0
    
    if model == "lin" or model == "LinearRegression":
        if "p_lin" in parameters:
            M = LinearRegression(**parameters["p_lin"],fit_intercept=False)
            M = M.fit(X_train,y_train)
        else:
            M = LinearRegression()
            M = M.fit(X_train,y_train)

    elif model == "rid" or model == "Ridge":
        if "p_rid" in parameters:
            M = Ridge(**parameters["p_rid"])
            M = M.fit(X_train,y_train)
        else:
            M = Ridge()
            M = M.fit(X_train,y_train)

    elif model == "las" or model == "Lasso":
        if "p_las" in parameters:
            M = Lasso(**parameters["p_las"])
            M = M.fit(X_train,y_train)
        else:
            M = Lasso()
            M = M.fit(X_train,y_train)

    elif model == "ela" or model == "ElasticNet":
        if "p_ela" in parameters:
            M = ElasticNet(**parameters["p_ela"])
            M = M.fit(X_train,y_train)
        else:
            M = ElasticNet()
            M = M.fit(X_train,y_train)

    elif model == "xgb" or model == "XGBRegressor":
        if "p_xgb" in parameters:
            M = XGBRegressor(**parameters["p_xgb"])
            M = M.fit(X_train,y_train)                
        else:
            M = XGBRegressor()
            M = M.fit(X_train,y_train)

    elif model == "svr" or model == "LinearSVR":
        if "p_svr" in parameters:
            M = LinearSVR(**parameters["p_svr"])
            M = M.fit(X_train,y_train)                
        else:
            M = LinearSVR()
            M = M.fit(X_train,y_train) 

    elif model == "knn" or model == "KNeighborsRegressor":
        if "p_knn" in parameters:
            M = KNeighborsRegressor(**parameters["p_knn"])
            M = M.fit(X_train,y_train)          
        else:
            M = KNeighborsRegressor()
            M = M.fit(X_train,y_train)

    elif model == "cat" or model == "CatBoostRegressor":
        if "p_cat" in parameters:
            M = CatBoostRegressor(**parameters["p_cat"])
            M = M.fit(X_train,y_train)                
        else:
            M = CatBoostRegressor(verbose=0)
            M = M.fit(X_train,y_train)

    elif model == "tre" or model == "DecisionTreeRegressor":
        if "p_tre" in parameters:
            M = DecisionTreeRegressor(**parameters["p_tre"],random_state=False)
            M = M.fit(X_train,y_train)                
        else:
            M = DecisionTreeRegressor(random_state=False)
            M = M.fit(X_train,y_train)

    else:
        print("The model is unknown")
        unknown = 1
        
    if unknown == 0:
      y_pred = M.predict(X_test)
    
    else:
      y_pred != (y_test == False)
   
    return y_pred

def plot_pred(y_train,y_test,y_train_pred,y_test_pred,index,name_model):
  """
  Plot the prediction of the model in order to compare with the real model

  name_model is a string where values can be:
  * LinearRegression for LinearRegression
  * Ridge for Ridge
  * Lasso for Lasso
  * ElasticNet for ElasticNet
  * XGBRegressor for XGBRegressor
  * LinearSVR for LinearSVR
  * KNeighborsRegressor for KNeighborsRegressor
  * CatBoostRegression for CatBoostRegression
  """
  plt.figure(figsize=(12,8))
  y_train = list(y_train)
  date_split = index[len(y_train)]
  y = y_train + list(y_test)
  y_p = list(y_train_pred) + list(y_test_pred)
  plt.plot(index,y,color='c')
  plt.plot(index,y_p,color='r',linestyle='dashed')
  plt.axvline(x = date_split, color = 'b')
  plt.title(f'Model: {name_model}')
  plt.show()

def plot_pred_detail(y_train,y_test,y_train_pred,y_test_pred,index,name_model,df_score, ic=True):
  """
  Plot the prediction of the model in order to compare with the real model

  name_model is a string where values can be:
  * LinearRegression for LinearRegression
  * Ridge for Ridge
  * Lasso for Lasso
  * ElasticNet for ElasticNet
  * XGBRegressor for XGBRegressor
  * LinearSVR for LinearSVR
  * KNeighborsRegressor for KNeighborsRegressor
  * CatBoostRegression for CatBoostRegression
  """
  fig, ax = plt.subplots(figsize=(18,8))
  r2 = df_score.loc[name_model,'r2']
  rmse = df_score.loc[name_model,'rmse']
  y_train = list(y_train)
  date_split = index[len(y_train)]
  y = y_train + list(y_test)
  y_p = list(y_train_pred) + list(y_test_pred)
  y_plot = ax.plot(index,y,color='c',label='historical values')
  y_pred_plot = ax.plot(index,y_p,color='r',linestyle='dashed',label='predicted values')
  ax.axvline(x = date_split, color = 'b')

  #if ic:
    #hypothesis that the errors are normally distributed; IC de 90%
    #ax.fill_between(index, y_p - 1.96*rmse, y_p + 1.96*rmse, color='r', alpha=0.1,label='IC90%')
  
  extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
  scores = (r"$R^2={:.4f}$" + "\n" + r"$RMSE={:.6f}$").format(r2,rmse)
  plt.legend([extra], [scores], loc='upper center')
  
  plt.xlabel("Date")
  plt.ylabel("DR")
  plt.title(f'Model: {name_model}')
  plt.show()

##### TIME SERIES FUNCTIONS #####

def acf_pacf(y,alpha=0.05,plot=True):
  """
  - Differencing the data while the adfuller test is rejected
  - Plotting the acf and pacf of a series 
  - Alpha is the threshold for the adf test
  - Returns the level of differenciation to use in order to have the stationarity of data
  """

  if type(y) != pd.core.series.Series:
    y = pd.Series(y)
  df = y.to_frame(name="y")
  d=0
  df.index = [i for i in range(y.shape[0])]
  test = adfuller(df.iloc[:,-1])

  while test[1]>=alpha and d<8:
    df = df.copy()
    df[f"y_diff{d+1}"] = df.iloc[:,-1].diff()
    test = adfuller(df.iloc[:,-1].dropna())
    d+=1

  print('\n'+'='*80)
  print(f'Test statistic for the series {df.columns[-1]}: {round(test[0],3)}')
  print(f'P-value for the series {df.columns[-1]}: {round(test[1],5)}')
  print("The data is potentially stationary. We reject the hypothesis of a unit root.")
  print(f"The level of differencing you need to use in order to have stationarity data is {d}.")
  print('='*80+'\n')

  if plot==True:
    if d != 0:
      fig, axes = plt.subplots(df.shape[1],3, sharex=True,figsize=(15,8*df.shape[1]))
      for i in range(df.shape[1]):
        y = df.iloc[:,i].dropna()
        axes[i,0].plot(y)
        axes[i,0].set_title(df.columns[i])
        plot_acf(y.values, ax=axes[i,1],bartlett_confint=True,alpha=0.05)
        axes[i,1].set_title('ACF') 
        plot_pacf(y.values,lags=y.shape[0]//2-1, ax=axes[i,2],alpha=0.05)
        axes[i,2].set_title('PACF') 
    else:
        fig, axes = plt.subplots(df.shape[1],3, sharex=True,figsize=(15,8))
        y = df.iloc[:,0].dropna()
        axes[0].plot(y)
        axes[0].set_title(df.columns[0])
        plot_acf(y.values, ax=axes[1],bartlett_confint=True,alpha=0.05)
        axes[1].set_title('ACF') 
        plot_pacf(y.values,lags=y.shape[0]//2-1, ax=axes[2],alpha=0.05)
        axes[2].set_title('PACF') 
    plt.show()

  return d

def find_arimax_params(y_train,exog=None,d=None,alpha=0.05,seasonal=False):
  """
  - Returns the best arima parameters for the data
  - You can choose the order of the arima model or let the model choose for you
  - You can choose the order of the seasonal arima model
  - Alpha is the threshold for the adf test
  """

  if d == None:
    d = acf_pacf(np.array(y_train),alpha=alpha,plot=False)

  if type(exog) == type(None):
    model = pm.auto_arima(y_train, d=d,seasonal=seasonal,stepwise=True,trace=True,error_action='ignore',suppress_warnings=True)
    print(model.summary())
  
  else:
    model = pm.auto_arima(y_train,exog,d=d,seasonal=seasonal,stepwise=True,trace=True,error_action='ignore',suppress_warnings=True)
    print(model.summary())

  # plot the residuals
  residuals = pd.DataFrame(model.resid())
  fig, axes = plt.subplots(1, 2, figsize=(15,5))
  axes[0].plot(residuals)
  axes[0].set_title('Residuals')
  plot_acf(residuals.values, ax=axes[1],bartlett_confint=True,alpha=0.05)
  axes[1].set_title('ACF')
  plt.show()

  # test for stationarity
  test = adfuller(residuals)
  print('\n'+ '='*80)
  print(f'Test statistic for the residuals: {round(test[0],3)}')
  print(f'P-value for the residuals: {round(test[1],5)}')
  if test[1]>=alpha:
    print("The residuals are not stationary. We accept the hypothesis of a unit root.")
  else:
    print("The residuals are potentially stationary. We reject the hypothesis of a unit root.")
    print(f"The level of differencing you need to use in order to have stationarity data is {d}.")
  print('='*80+'\n')

  return model

def plot_arimax_fit(model,y_train,y_test,exog_train=None,exog_test=None):
    """
    - Plot the fit and the prediction of the arimax model (with exogenous data or not)
    - A horizontal bar is plotted to show the test set
    """

    y_total = pd.concat([y_train,y_test])

    if type(exog_train)==type(None) and type(exog_test)==type(None):
      y_fit = model.predict(y_train.shape[0])
      y_fit.index = y_train.index
      y_predict,y_conf = model.predict(y_test.shape[0],return_conf_int=True)
    else:
      y_fit = model.predict(y_train.shape[0],exog_train)
      y_fit.index = y_train.index
      y_predict,y_conf = model.predict(y_test.shape[0],exog_test,return_conf_int=True)

    fig, axes = plt.subplots(1, 1, figsize=(15,5))
    axes.plot(y_total.index, y_total.values)

    # plot the fit 
    y_fit_predict = pd.concat([pd.Series(y_fit,index=y_train.index),pd.Series(y_predict,index=y_test.index)])
    axes.axvline(y_test.index[0],color='r')
    axes.plot(y_fit_predict.index, y_fit_predict.values)
    
    # plot confidence interval
    y_conf = pd.DataFrame(y_conf,index=y_test.index,columns=['lower','upper'])
    axes.fill_between(y_conf.index, y_conf['lower'], y_conf['upper'], alpha=0.15, color='r')
    
    plt.legend(['Original','Fit + Prediction'])
    plt.show()