import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from scipy.fftpack import fft, ifft, rfft
from datetime import timedelta
from joblib import dump, load
import pickle

testingData=pd.read_csv('test.csv',header=None)

def extractNoMealFeatures(noMealData):
    indexRemoveNonMeal=noMealData.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    noMealData_cleaned=noMealData.drop(noMealData.index[indexRemoveNonMeal]).reset_index().drop(columns='index')
    noMealData_cleaned=noMealData_cleaned.interpolate(method='linear',axis=1)
    indexDrop=noMealData_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    noMealData_cleaned=noMealData_cleaned.drop(noMealData_cleaned.index[indexDrop]).reset_index().drop(columns='index')
    nonMealMatrix=pd.DataFrame()
    noMealData_cleaned['tau_time']=(24 - noMealData_cleaned.iloc[:,0:19].idxmax(axis=1))*5
    noMealData_cleaned['difference_in_glucose_normalized']=(noMealData_cleaned.iloc[:,0:19].max(axis=1)-noMealData_cleaned.iloc[:,24]) / (noMealData_cleaned.iloc[:,24])
    power_1 = []
    index_1 = []
    power_2 = []
    index_2 = []
    for i in range(len(noMealData_cleaned)):
        arr=abs(rfft(noMealData_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sortedArr=abs(rfft(noMealData_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sortedArr.sort()
        power_1.append(sortedArr[-2])
        power_2.append(sortedArr[-3])
        index_1.append(arr.index(sortedArr[-2]))
        index_2.append(arr.index(sortedArr[-3]))
    nonMealMatrix['tau_time']=noMealData_cleaned['tau_time']
    nonMealMatrix['difference_in_glucose_normalized']=noMealData_cleaned['difference_in_glucose_normalized']
    nonMealMatrix['power_first_max']=power_1
    nonMealMatrix['power_second_max']=power_2
    nonMealMatrix['index_first_max']=index_1
    nonMealMatrix['index_second_max']=index_2
    firDiffData=[]
    secDiffData=[]
    for i in range(len(noMealData_cleaned)):
        firDiffData.append(np.diff(noMealData_cleaned.iloc[:,0:24].iloc[i].tolist()).max())
        secDiffData.append(np.diff(np.diff(noMealData_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
    nonMealMatrix['1stDifferential']=firDiffData
    nonMealMatrix['2ndDifferential']=secDiffData
    return nonMealMatrix

dataset=extractNoMealFeatures(testingData)

from joblib import dump, load
with open('DecisionTreeClassifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    prediction = pickle_file.predict(dataset)    
    pre_trained.close()

pd.DataFrame(prediction).to_csv('Result.csv',index=False,header=False)