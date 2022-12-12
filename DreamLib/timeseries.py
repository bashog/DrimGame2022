import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.metrics import mean_squared_error as mse

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
    

def plot_forecasts(forecasts, title, figsize=(8, 12)):
    """
    - Plot the forecasts and the residuals of the model
    """

    x = np.arange(y_train.shape[0] + forecasts.shape[0])

    fig, axes = plt.subplots(2, 1, sharex=False, figsize=figsize)

    # Plot the forecasts
    axes[0].plot(x[:y_train.shape[0]], y_train, c='b')
    axes[0].plot(x[y_train.shape[0]:], forecasts, c='g')
    axes[0].set_xlabel(f'Sunspots (RMSE={np.sqrt(mse(y_test, forecasts)):.3f})')
    axes[0].set_title(title)

    # Plot the residuals
    resid = y_test - forecasts
    _, p = normaltest(resid)
    axes[1].hist(resid, bins=15)
    axes[1].axvline(0, linestyle='--', c='r')
    axes[1].set_title(f'Residuals (p={p:.3f})')

    plt.tight_layout()
    plt.show()