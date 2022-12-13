##### LIBRAIRIES #####

# Models
from pickle import FALSE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# Grid Search
from sklearn.model_selection import GridSearchCV

# Data processing
import pandas as pd
import numpy as np

# Plot
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'


  
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

def plot_pred_detail(y_train,y_test,y_train_pred,y_test_pred,index,name_model,df_score,period=None,y_validation=None,ic=True):
  """
  Plot the prediction of the model in order to compare with the real model
  If y_validation is None this do not represent the value predicte without knowing the real value 
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
  # r2 = df_score.loc[name_model,'r2']
  rmse = df_score.loc[name_model,'rmse']
  y_train = list(y_train)
  date_split = index[len(y_train)]
  if type(y_validation) != type(None):
    y = y_train + list(y_test) + list(np.repeat(np.nan,y_validation.shape[0]))
    y_p = list(y_train_pred) + list(y_test_pred) + list(y_validation)
    ax.axvline(x = index[-period//3], color = 'r')
  else: 
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
  scores = (r"$RMSE={:.6f}$").format(rmse)
  plt.legend([extra], [scores], loc='upper center')

  plt.xlabel("Date")
  plt.ylabel("DR")
  plt.title(f'Model: {name_model}')
  plt.show()