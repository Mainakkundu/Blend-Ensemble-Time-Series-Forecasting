# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:34:32 2019

@author: mainak.kundu
"""




#Created on Mon Jun 17 15:49:31 2019

#@author: mainak.kundu


import pandas as pd 
import numpy as np 
import dask as dd
import numpy as np 
import os 
import matplotlib.pyplot as plt
plt.style.use('classic')
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns 
import keras 
from keras.layers import Activation, Dense
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import sys 



p1 = str.replace(sys.argv[1],"#","//") ## input 
p2 = str.replace(sys.argv[2],"#","//") ## output
#p9 = str.replace(sys.argv[3],'#',"//") ## hyperdata 
p9 = sys.argv[9].split(",") ### for sales booster 
p3 = sys.argv[3].split(",")
p4 = sys.argv[4].split(",")
p5 = sys.argv[5].split(",")
p6 = sys.argv[6]
p7 = sys.argv[7]
p8 = sys.argv[8]
print(p3)
print(p4)
#exit()
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)



'''
EXTRACT TIME RELATED FEATURES (WeekNo,Mnth,Year)------------
'''
def TimeSorting(df):
    '''
    Extract 3 Features Year,Month,WeekNo
    '''
    df_1['TRDNG_WK_END_DT'] = pd.to_datetime(df_1['TRDNG_WK_END_DT'],infer_datetime_format=True)
    df= df.sort_values('TRDNG_WK_END_DT')
    df['year'] = pd.DatetimeIndex(df['TRDNG_WK_END_DT']).year
    df['Month'] = pd.DatetimeIndex(df['TRDNG_WK_END_DT']).month
    df['WeekNo'] = df['TRDNG_WK_END_DT'].dt.week
    return df 






import math 
def time_based_split(riyad_fl_df):
    riyad_fl_df.sort_values(['TRDNG_WK_END_DT'],ascending=[True], inplace=True)
    row_shape_train = math.ceil(riyad_fl_df.shape[0]*0.5)
    row_shape_test = math.floor(riyad_fl_df.shape[0]*0.5)
    train = riyad_fl_df.head(row_shape_train)
    test = riyad_fl_df.tail(row_shape_test)
    return train,test

def time_based_split_unseen(test):
    '''
    BASICALLY NO NEED WHEN WE HAVE ROUT
    '''
    test.sort_values(['TRDNG_WK_END_DT'],ascending=[True], inplace=True)
    row_shape_test = math.ceil(test.shape[0]*0.2)
    un_test = test.tail(row_shape_test)
    return un_test

def clean_up(df):
    for a,b,c in zip(df['ARIMA'],df['LREG'],df['ENSEMBLE_FCST']):
        np.where(c>b and c>a,a+b+c/3,c)
    return df





def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    smape = SMAPE(y_true,y_pred)
    print('Mean Squared Error,','MAPE,','RMSE','SMAPE')
    return round(mse, 3), round(mape, 3), round(rmse, 3), round(smape,3)






print('---- BASE LEARNERS MODEL----')
def base_learner_models():
    '''
    All Base Model which you want to use
    '''
    SEED=1234
    #svc = SVR(gamma='scale',kernel='rbf')
    knn = KNeighborsRegressor(n_neighbors=3)
    lr  = LinearRegression()
    #gb  = GradientBoostingRegressor(n_estimators=10, random_state=SEED)
    rf  = RandomForestRegressor(n_estimators=10, max_features=3, random_state=SEED)
    extra = best_model_extratree
    #fnn = mlp
    models = {#'svm': svc,
              'knn': knn,
              'random forest': rf,
              #'gbm': gb,
              'linear': lr,
              'extra':extra,
              #'fnn':fnn
              }

    return models
#base_learners = base_learner_models() ## grabing my base learners 

print('---Base Learners learning phase started ----')
def base_learner_training_predicting(base_learners,X_train,X_test,y_train,X_un_test):
    P = np.zeros((X_test.shape[0], len(base_learners))) ## 50% of data 
    P = pd.DataFrame(P)
    Q = np.zeros((X_un_test.shape[0], len(base_learners))) ## Rout data 
    Q = pd.DataFrame(Q)
    
    print("---Fitting models---")
    cols = list()
    for i, (name, m) in enumerate(base_learners.items()):
        #print("%s..." % name, end=" ", flush=False)
        m.fit(X_train.fillna(0), y_train)
        P.iloc[:, i] = m.predict(X_test.fillna(0))
        Q.iloc[:, i] = m.predict(X_un_test.fillna(0))
        cols.append(name)
        print("done")
    P.columns = cols
    Q.columns = cols
    return P,Q



'''
ERROR CAPTURED
'''
def deviations_calculation(train, method_of_fcst='LREG',dependent_var = 'RTL_QTY'):
    train['E'+'_'+method_of_fcst] = train[dependent_var] - train[method_of_fcst]
    return train 


'''
TRAIN,TEST SPLIT FOR ERRROR MODELING 
'''
def train_test_splt_error_modeling(train,test,method_of_fcst='linear'):
    X_train = train.drop(['E_'+method_of_fcst],axis=1)
    X_train = X_train[method_of_fcst]
    X_test = test[method_of_fcst]
    y_train = train['E_'+method_of_fcst]
    print('--Shape of data--')
    print(X_train.shape,y_train.shape)
    return X_train,y_train,X_test 


from sklearn.model_selection import cross_val_score
from sklearn import  metrics

def modf_ensemble_engine(alg,X_train,y_train,X_test):
    X_train.fillna(0,inplace=True)
    y_train.fillna(0,inplace=True)
    X_test.fillna(0,inplace=True)
    X_tr = np.array(X_train.values).reshape(-1,1)
    y_tr = np.array(y_train.values)
    X_ts = np.array(X_test.values).reshape(-1,1)
    alg.fit(X_tr,y_train)
    dtrain_prd = alg.predict(X_tr)
    print('---Cross Validation Start---')
    cv_score = cross_val_score(alg, X_tr,y_tr,cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    print('--Model Report---')
    print("\nModel Report")
    print(np.sqrt(metrics.mean_squared_error(y_train, dtrain_prd)))
    print('--Prediction Stage--')
    dtest_prd = alg.predict(X_ts)
    return dtest_prd,alg





def ensemble_engine_for_error_modeling(X_train,y_train,X_test):
    X_train.fillna(0,inplace=True)
    y_train.fillna(0,inplace=True)
    X_test.fillna(0,inplace=True)
    best_model_extratree = ExtraTreesRegressor()
    X_tr = np.array(X_train.values).reshape(-1,1)
    y_tr = np.array(y_train.values)
    X_ts = np.array(X_test.values).reshape(-1,1)
    best_model_extratree.fit(X_tr,y_tr)
    print('---R-sqaure--')
    print(r2_score(y_train,best_model_extratree.predict(X_tr)))
    pred_ensemble_forecast = best_model_extratree.predict(X_ts)
    return pred_ensemble_forecast,best_model_extratree




def sigmoid(x):
  return 1 / (1 + math.exp(-x))



if __name__== '__main__':

    ## Path Declration 
    path = r'D:\PROJECTS\IM\ENSEMBLE TIME SERIES\HC_data\HC_NEW_ENSEMBLE_DATA\VERSION_2_TESTING'
    os.chdir(path)
    import sys
    usr = p6
    cncpt = p7
    dept = p8
    logs = open(usr+'_'+cncpt+'_'+dept+'_'+'ENSEMBLE_LOGS.txt','w')
    sys.stdout = logs
    
    ## Territory which we want Forecast (args)
    TERR_LIST = ['United Arab Emirates','Jeddah - KSA','Lebanon', 'Egypt', 'Oman','Riyadh - KSA', 'Bahrain', 'Jordan','Qatar', 'Dammam - KSA','Kuwait']
      
    
    #df = pd.read_csv('IM_BS_ALL_ENSB_HADS_n.txt') ## Training data 
    #rout = pd.read_csv('IM_BS_ALL_ENSB_LADS_n.txt') ## Testing Data 
    HISTDATA_PATH = p1 ## path for HISTORICAL SALES
    df = read.csv(HISTDATA_PATH)
    LEADATA_PATH = p2  ## ROUT path 
    df = df[df['STND_TRRTRY_NM'].isin(TERR_LIST)]
    rout = pd.read_csv(LEADATA_PATH)
    rout = rout[rout['STND_TRRTRY_NM'].isin(TERR_LIST)]
    
    key = [v for v in df['KEY'].unique()]   
    #key = key[0:2]
    print (len(key))
    ## For doing Randomized Grid Search for wrapper 
    param_grid = {"n_estimators": [10, 50, 75, 100, 150],"max_features": ["auto", "sqrt", "log2"],"max_depth": [50, 100,150, 200, 250]}
     
    ## Indipendent variable list which Concept specifics (args)
    FEATURES_SET_BASELEARNER = ['LREG_FCST','ARIMA_FCST','ETS_FCST','AVG_FCST','FLG_BTS','FLG_NATIONAL_DAY','FLG_SALE','FLG_RAMADAN','FLG_EID2','FLG_MSOP','FLG_MSOS','FLG_MSOW','FLG_SEOSS','FLG_SPOSS','FLG_WEOSS'] ##args              

    ## Sales Booster (which you felt might be giving peak)
    SALES_BOOSTER = ['FLG_BTS','FLG_NATIONAL_DAY','FLG_SALE','FLG_RAMADAN','FLG_EID2','FLG_MSOP','FLG_MSOS','FLG_MSOW','FLG_SEOSS','FLG_SPOSS','FLG_WEOSS'] ## args 
    DEPENDENT_VAR = 'RTL_QTY'
    FEATURES_SET_BASELEARNER = FEATURES_SET_BASELEARNER+SALES_BOOSTER

    resultF    = pd.DataFrame() # result dataframe 
    for i in key:
        print(i)
        df_1 = df[(df['KEY'] == i)]
        df_1['TRDNG_WK_END_DT'] = pd.to_datetime(df_1['TRDNG_WK_END_DT'],infer_datetime_format=True)
        df_1 = TimeSorting(df_1)
        
        #print('--Before Outlier Treatment---')
        #print(df_1['RTL_QTY'].max())
        #print(df_1['RTL_QTY'].min())
        #print('-----After Outlier Treatment----')
        #percentiles = df_1['RTL_QTY'].quantile([0.01,0.99]).values
        #df_1['RTL_QTY'][df_1['RTL_QTY'] <= percentiles[0]] = percentiles[0]
        #df_1['RTL_QTY'][df_1['RTL_QTY']  >= percentiles[1]] = percentiles[1]
        df_1['ENSEMBLE_FCST_FLG'] = np.where((df_1['LREG_FLG']+df_1['ARIMA_FLG']+df_1['ETS_FLG'] >= 3),'stable','unstable')
        
       
        try:
            df_1.drop([df_1['ENSEMBLE_FCST_FLG']==3],inplace=True)
        except:
            print('All Forecasts are good')
            
        
        train,test = time_based_split(df_1)
        print('--Train,test shape---')
        print(train.shape,test.shape)
        un_test = rout[(rout['KEY'] == i)] ## filter KEY
        un_test['TRDNG_WK_END_DT'] = pd.to_datetime(rout['TRDNG_WK_END_DT'],infer_datetime_format=True)
        un_test['ENSEMBLE_FCST_FLG'] = np.where((un_test['LREG_FLG']+un_test['ARIMA_FLG']+un_test['ETS_FLG'] >= 3),'stable','unstable')
          

        #un_test = time_based_split_unseen(test) ### basically rout file (directly read.csv to read that file make sure 330(pdtimsestamp conversion) TimeSorting should apply)
        print('---Rout data shape--')
        print(un_test.shape)
        #FEATURES_SET_BASELEARNER = 'LREG_FCST','ARIMA_FCST','ETS_FCST','AVG_FCST','FLG_SU_EOSS', 'FLG_WN_EOSS', 'FLG_SP_EOSS','FLG_WN_EOSS', 'FLG_SP_EOSS','FLG_RAMADAN','FLG_EID','FLG_MID_SSN_OFFER' ##args              

        print('---1st Phase Started----')
        X_train = train.drop(['RTL_QTY'],axis=1)
        y_train = train['RTL_QTY']
        X_test = test.drop(['RTL_QTY'],axis=1)
        y_test = test['RTL_QTY']
        X_train = X_train[FEATURES_SET_BASELEARNER]
        X_test = X_test[FEATURES_SET_BASELEARNER]
        X_un_test = un_test[FEATURES_SET_BASELEARNER]
        X_train.fillna(0,inplace=True)
        X_test.fillna(0,inplace=True)
        X_un_test.fillna(0,inplace=True)
        print('---Orginal data Set shape----')
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        
        

        print('-----Mean Encoding for Sales Booster-----')
        for sale_i in SALES_BOOSTER:
            print(sale_i)
            ordered_labels =df_1.groupby([sale_i])[DEPENDENT_VAR].mean().to_dict()
            print(ordered_labels)
            X_train[sale_i] = X_train[sale_i].replace(ordered_labels)
            X_test[sale_i] = X_train[sale_i].replace(ordered_labels)
            X_un_test[sale_i] = X_un_test[sale_i].replace(ordered_labels)
        print('Shape after Train_50% Mean Encoding:{}'.format(X_train.shape))
        print('Shape after Train_50% Mean Encoding:{}'.format(X_test.shape))
        print('Shape after Rout Mean Encoding:{}'.format(X_un_test.shape))
        
        print('------ Base Learner Modeling Started------')
        extraTree = RandomForestRegressor()
        grid_search = RandomizedSearchCV(extraTree, param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train.fillna(0), y_train)
        print("Parameters of best Regressor : {}".format(grid_search.best_params_))
        best_model_extratree = grid_search.best_estimator_
        best_model_extratree.fit(X_train,y_train)
        base_learners = base_learner_models()
        X_train_new,X_test_new = base_learner_training_predicting(base_learners,X_train,X_test,y_train,X_un_test)
        print('--------1st Phase Completed------')

        print('------ 2nd Phase Started for Error Modeling-----')
        error_df_train = pd.DataFrame()
        error_df_train['linear'] = X_train_new['linear'].values
        error_df_train['knn'] = X_train_new['knn'].values
        error_df_train['extra'] = X_train_new['extra'].values
        error_df_train['random forest'] = X_train_new['random forest'].values
        error_df_train['RTL_QTY'] = test['RTL_QTY'].values

        error_df_test = pd.DataFrame()
        error_df_test['linear'] = X_test_new['linear'].values
        error_df_test['knn'] = X_test_new['knn'].values
        error_df_test['extra'] = X_test_new['extra'].values
        error_df_test['random forest'] = X_test_new['random forest'].values
        print('----Check the train have missing values')
        print(error_df_train.isnull().sum())
        print(error_df_train.shape)
        print('----Check the test have missing values')
        print(error_df_test.isnull().sum())
        print(error_df_test.shape)
        test_ERROR = error_df_test
        FORECAST_LIST = ['linear','knn','extra','random forest']
        for j in FORECAST_LIST:
            train_ERROR = deviations_calculation(error_df_train,method_of_fcst=j,dependent_var='RTL_QTY')
        print('Shape of train,test data')
        print(train_ERROR.shape,test_ERROR.shape)
        X_train_linear,y_train_linear,X_test_linear = train_test_splt_error_modeling(train_ERROR,test_ERROR,method_of_fcst='linear')
        X_train_knn,y_train_knn,X_test_knn = train_test_splt_error_modeling(train_ERROR,test_ERROR,method_of_fcst='knn')
        X_train_extra,y_train_extra,X_test_extra = train_test_splt_error_modeling(train_ERROR,test_ERROR,method_of_fcst='extra')
        X_train_random_forest,y_train_random_forest,X_test_random_forest = train_test_splt_error_modeling(train_ERROR,test_ERROR,method_of_fcst='random forest')
        alg = LinearRegression(normalize=True)
        ensemble_linear_pred, m_linear = modf_ensemble_engine(alg,X_train_linear,y_train_linear,X_test_linear)
        ensemble_knn_pred,m_knn = modf_ensemble_engine(alg,X_train_knn,y_train_knn,X_test_knn)
        ensemble_extra_pred,m_extra = modf_ensemble_engine(alg,X_train_extra,y_train_extra,X_test_extra)
        ensemble_rf,m_rf = modf_ensemble_engine(alg,X_train_random_forest,y_train_random_forest,X_test_random_forest)
        print('---Weight matrix started ----')
        print(i)
        #v = i
        weights = pd.DataFrame()
        weights['MKTNG_FLG'] = un_test[SALES_BOOSTER].sum(axis=1)
        weights['KEY'] = un_test['KEY']
        weights['STND_TRRTRY_NM'] = un_test['STND_TRRTRY_NM']
        weights['TRDNG_WK_END_DT'] = un_test['TRDNG_WK_END_DT']
        weights['ARIMA_FCST'] = un_test['ARIMA_FCST']
        weights['LREG_FCST'] = un_test['LREG_FCST']
        weights['ETS_FCST'] = un_test['ETS_FCST']
        weights['AVG_FCST'] = un_test['AVG_FCST']
        #weights['RTL_QTY'] = un_test['RTL_QTY'].values
        weights['BLLINEAR'] = test_ERROR['linear'].values
        weights['BLKNN']  = test_ERROR['knn'].values
        weights['BLEXTRA'] = test_ERROR['extra'].values
        weights['BLRF'] = test_ERROR['random forest'].values
        weights['ENS_C_sum'] = weights['ARIMA_FCST'] + weights['LREG_FCST']
        weights['ENS_C'] = weights['ENS_C_sum']/2
        weights['E_HAT_BLLINEAR'] = np.abs(ensemble_linear_pred)
        weights['E_HAT_BLKNN'] =np.abs(ensemble_knn_pred)
        weights['E_HAT_BLEXTRA'] = np.abs(ensemble_extra_pred)
        weights['E_HAT_BLRF'] = np.abs(ensemble_rf)
        weights['SIG_BLLINEAR'] = weights['E_HAT_BLLINEAR'].apply(sigmoid)
        weights['SIG_BLKNN'] = weights['E_HAT_BLKNN'].apply(sigmoid)
        weights['SIG_BLEXTRA'] = weights['E_HAT_BLEXTRA'].apply(sigmoid)
        weights['SIG_BLRF'] = weights['E_HAT_BLRF'].apply(sigmoid)
        weights['W_BLLINEAR']  = weights['SIG_BLLINEAR']/(weights['SIG_BLLINEAR']+weights['SIG_BLKNN']+weights['SIG_BLEXTRA']+weights['SIG_BLRF'])
        weights['W_BLKNN']  = weights['SIG_BLKNN']/(weights['SIG_BLLINEAR']+weights['SIG_BLKNN']+weights['SIG_BLEXTRA']+weights['SIG_BLRF'])
        weights['W_BLEXTRA']  = weights['SIG_BLEXTRA']/(weights['SIG_BLLINEAR']+weights['SIG_BLKNN']+weights['SIG_BLEXTRA']+weights['SIG_BLRF'])
        weights['W_BLRF']  = weights['SIG_BLRF']/(weights['SIG_BLLINEAR']+weights['SIG_BLKNN']+weights['SIG_BLEXTRA']+weights['SIG_BLRF'])
        weights['ENSEMBLE_FCST'] = (weights['W_BLLINEAR']*weights['SIG_BLLINEAR'])+(weights['W_BLEXTRA']*weights['BLEXTRA'])+(weights['BLRF']*weights['W_BLRF'])+(weights['BLKNN']*weights['W_BLKNN'])
        weights['ENSEMBLE_FCST_FLG'] = un_test['ENSEMBLE_FCST_FLG']
        nw_w = []
        for ei,ar,lre,ei_ad,mk in zip(weights['ENSEMBLE_FCST'],weights['ARIMA_FCST'],weights['LREG_FCST'],weights['ENS_C'],weights['MKTNG_FLG']):
            if (ei < lre) and (ei < ar):
                #print(ei)
                nw_w.append(ei_ad)
            elif (ei > lre) and (ei < ar):
                nw_w.append(ei)
            elif (ei < lre) and (ei > ar):
                nw_w.append(ei)
            elif (lre < 0) and (ar <0):
                nw_w.append(ei_ad)
            elif mk>1:
                l = [lre,ar]
                v = np.max(l)
                v = v + v*0.8
                nw_w.append(v)
            elif (mk>0) and (ei < lre) and (ei < ar):
                l = [lre,ar]
                nw_w.append(np.max(l))
            
            else:
                nw_w.append(ei_ad)
        
        weights['ENSEMBLE_FCST_ADJ'] = np.abs(nw_w)        
            
        resultF = pd.concat([resultF,weights])
        cols = ['STND_TRRTRY_NM','KEY','TRDNG_WK_END_DT','ENSEMBLE_FCST','ENSEMBLE_FCST_ADJ','ARIMA_FCST','LREG_FCST','ETS_FCST','AVG_FCST']
        resultF = resultF[cols]
        resultF.to_csv(usr+'_'+cncpt+'_'+dept+'_'+'ENSEMBLE_PYOUT_WO_OUTLIERS_REMOVING_BAD_DATA.csv')
        print('--------DONE-----------------------')




    
   











       

    
    





