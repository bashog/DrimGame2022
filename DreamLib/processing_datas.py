import pandas as pd

# Normalize
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# Train Test Split
from sklearn.model_selection import train_test_split

##### PROCESSING FUNCTIONS #####

def clean_data(data:pd.DataFrame,start:int,period:int,chronique,col_used:None,split=0.2,norm="MinMax"):
  """
  - Returns the data normalized and prepared for t -> X_train,X_test,y_train,y_test,X_validation in the case of a non null split else just return the all DataFrame
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
  - Period
  Choose the type of data you want to use (12, 24, or 36 months)
  """
  # Data processing
  X = data[data.CHRONIQUE == chronique]
  X = X.iloc[start:,]
  X = X.set_index(X.TRIMESTRE)
  X["DR"] = X["DR"].shift(-period//3)
  Y = X.DR.iloc[:-period//3]
  if col_used != None:
    X = X[col_used]
  else:
      X = X.drop(columns=['CD_TY_CLI_RCI_2','CD_ETA_CIV_2','CD_MOD_HABI_2','CD_PROF_3','CD_QUAL_VEH_2'])
  X = X.drop(columns=["DR","TRIMESTRE","CHRONIQUE"])
  X_validation = X.iloc[-period//3:,:]
  X = X.iloc[:-period//3,:]

  # Train Split
  if split == 0:
    X_train, X_test, y_train, y_test = X.dropna(axis=1),X.dropna(axis=1),Y,Y
  
  else:
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=split,shuffle=False)
  index_train = X_train.index
  index_test = X_test.index

  # Normalization
  if norm == "Not":
    if split == 0:
      return X_train,y_train
    else:
      return X_train, X_test, y_train, y_test,X_validation
  
  elif norm == 'MinMax':
    scaler_train = MinMaxScaler()
    scaler_test = MinMaxScaler()
    scaler_validation = MinMaxScaler()
    X_train = pd.DataFrame(scaler_train.fit_transform(X_train),index=index_train,columns=X_train.columns)
    X_test = pd.DataFrame(scaler_test.fit_transform(X_test),index=index_test,columns=X_test.columns)
    X_validation = pd.DataFrame(scaler_validation.fit_transform(X_validation),index=index_test,columns=X_validation.columns)

  elif norm == 'StdSca':
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    scaler_validation = StandardScaler()
    X_train = pd.DataFrame(scaler_train.fit_transform(X_train),index=index_train,columns=X_train.columns)
    X_test = pd.DataFrame(scaler_test.fit_transform(X_test),index=index_test,columns=X_test.columns)
    X_validation = pd.DataFrame(scaler_validation.fit_transform(X_validation),index=index_test,columns=X_validation.columns)
  
  else:
    print("This norm doesn't exist")

  if split == 0:
    return X_train,y_train
  
  else:
    return X_train, X_test, y_train, y_test,X_validation