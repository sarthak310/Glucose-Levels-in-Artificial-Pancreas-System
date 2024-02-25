import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from scipy.fftpack import fft, ifft, rfft
from datetime import timedelta
from joblib import dump, load

insulinData=pd.read_csv('InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
cgmData=pd.read_csv('CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])

insulinData['date_time_stamp']=pd.to_datetime(insulinData['Date'] + ' ' + insulinData['Time'])
cgmData['date_time_stamp']=pd.to_datetime(cgmData['Date'] + ' ' + cgmData['Time'])

insulinData_1=pd.read_csv('Insulin_patient2.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
cgmData_1=pd.read_csv('CGM_patient2.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])

insulinData_1['date_time_stamp']=pd.to_datetime(insulinData_1['Date'] + ' ' + insulinData_1['Time'])
cgmData_1['date_time_stamp']=pd.to_datetime(cgmData_1['Date'] + ' ' + cgmData_1['Time'])



# EXTRACTION OF MEAL DATA:


def extractMealData(insulinData,cgmData,dateidentifier):
    insulinDataCopy_1=insulinData.copy()
    insulinDataCopy_1=insulinDataCopy_1.set_index('date_time_stamp')
    tm2hr30min=insulinDataCopy_1.sort_values(by='date_time_stamp',ascending=True).dropna().reset_index()
    tm2hr30min['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    tm2hr30min=tm2hr30min.dropna()
    tm2hr30min=tm2hr30min.reset_index().drop(columns='index')
    validTimes_1=[]
    val=0
    for idx, i in enumerate(tm2hr30min['date_time_stamp']):
        try:
            val=(tm2hr30min['date_time_stamp'][idx + 1] - i).seconds / 60.0
            if val >= 120:
                validTimes_1.append(i)
        except KeyError:
            break
    list_1=[]
    if dateidentifier==1:
        for idx, i in enumerate(validTimes_1):
            begin=pd.to_datetime(i - timedelta(minutes = 30))
            end=pd.to_datetime(i + timedelta(minutes = 120))
            getDate=i.date().strftime('%#m/%#d/%Y')
            list_1.append(cgmData.loc[cgmData['Date']==getDate].set_index('date_time_stamp').between_time(start_time=begin.strftime('%#H:%#M:%#S'),end_time=end.strftime('%#H:%#M:%#S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list_1)
    else:
        for idx, i in enumerate(validTimes_1):
            begin=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            getDate=i.date().strftime('%Y-%m-%d')
            list_1.append(cgmData.loc[cgmData['Date']==getDate].set_index('date_time_stamp').between_time(start_time=begin.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list_1)

mealData=extractMealData(insulinData,cgmData,1)
mealData_1=extractMealData(insulinData_1,cgmData_1,2)
mealData=mealData.iloc[:,0:30]
mealData_1=mealData_1.iloc[:,0:30]



# EXTRACTION OF NON-MEAL DATA:


def extractNoMealData(insulinData,cgmData):
    insulinDataCopy_2=insulinData.copy()
    tm2hr=insulinDataCopy_2.sort_values(by='date_time_stamp',ascending=True).replace(0.0,np.nan).dropna().copy()
    tm2hr=tm2hr.reset_index().drop(columns='index')
    validTimes_2=[]
    for idx, i in enumerate(tm2hr['date_time_stamp']):
        try:
            val=(tm2hr['date_time_stamp'][idx + 1] - i).seconds // 3600
            if val >=4:
                validTimes_2.append(i)
        except KeyError:
            break
    list_2=[]
    for idx, i in enumerate(validTimes_2):
        iterDataset=1
        try:
            length=len(cgmData.loc[(cgmData['date_time_stamp']>=validTimes_2[idx] + pd.Timedelta(hours=2))&(cgmData['date_time_stamp']<validTimes_2[idx + 1])]) // 24
            while (iterDataset<=length):
                if iterDataset==1:
                    list_2.append(cgmData.loc[(cgmData['date_time_stamp']>=validTimes_2[idx] + pd.Timedelta(hours=2))&(cgmData['date_time_stamp']<validTimes_2[idx + 1])]['Sensor Glucose (mg/dL)'][:iterDataset*24].values.tolist())
                    iterDataset+=1
                else:
                    list_2.append(cgmData.loc[(cgmData['date_time_stamp']>=validTimes_2[idx] + pd.Timedelta(hours=2))&(cgmData['date_time_stamp']<validTimes_2[idx + 1])]['Sensor Glucose (mg/dL)'][(iterDataset - 1)*24:(iterDataset)*24].values.tolist())
                    iterDataset+=1
        except IndexError:
            break
    return pd.DataFrame(list_2)

noMealData=extractNoMealData(insulinData,cgmData)
noMealData_1=extractNoMealData(insulinData_1,cgmData_1)



# FEATURE EXTRACTION FROM MEAL DATA....FEATURES CONSIDERED:
# 1 - Fast Fourier Transform (FFT) of second and third highest power and indices
# 2 - tau_time: time interval bet max and min glucose level
# 3 - First-order differential of glucose lvl w.r.t time
# 4 - Second-order differential of glucose lvl w.r.t time
# 5 - difference_in_glucose_normalized: change in glucose bet min and max lvls over min glucose lvl


def extractMealFeatures(mealData):
    index=mealData.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    mealData_cleaned=mealData.drop(mealData.index[index]).reset_index().drop(columns='index')
    mealData_cleaned=mealData_cleaned.interpolate(method='linear',axis=1)
    indexDrop=mealData_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    mealData_cleaned=mealData_cleaned.drop(mealData.index[indexDrop]).reset_index().drop(columns='index')
    mealData_cleaned['tau_time']=(mealData_cleaned.iloc[:,22:25].idxmin(axis=1)-mealData_cleaned.iloc[:,5:19].idxmax(axis=1))*5
    mealData_cleaned['difference_in_glucose_normalized']=(mealData_cleaned.iloc[:,5:19].max(axis=1) - mealData_cleaned.iloc[:,22:25].min(axis=1)) / (mealData_cleaned.iloc[:,22:25].min(axis=1))
    mealData_cleaned=mealData_cleaned.dropna().reset_index().drop(columns='index')
    power_1 = []
    index_1 = []
    power_2 = []
    index_2 = []
    for i in range(len(mealData_cleaned)):
        arr=abs(rfft(mealData_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sortedArr=abs(rfft(mealData_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sortedArr.sort()
        power_1.append(sortedArr[-2])
        power_2.append(sortedArr[-3])
        index_1.append(arr.index(sortedArr[-2]))
        index_2.append(arr.index(sortedArr[-3]))
    mealMatrix=pd.DataFrame()
    mealMatrix['tau_time']=mealData_cleaned['tau_time']
    mealMatrix['difference_in_glucose_normalized']=mealData_cleaned['difference_in_glucose_normalized']
    mealMatrix['power_first_max']=power_1
    mealMatrix['power_second_max']=power_2
    mealMatrix['index_first_max']=index_1
    mealMatrix['index_second_max']=index_2
    tm=mealData_cleaned.iloc[:,22:25].idxmin(axis=1)
    maximum=mealData_cleaned.iloc[:,5:19].idxmax(axis=1)
    firDiffData=[]
    secDiffData=[]
    sd=[]
    for i in range(len(mealData_cleaned)):
        firDiffData.append(np.diff(mealData_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()).max())
        secDiffData.append(np.diff(np.diff(mealData_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        sd.append(np.std(mealData_cleaned.iloc[i]))
    mealMatrix['1stDifferential']=firDiffData
    mealMatrix['2ndDifferential']=secDiffData
    return mealMatrix

mealMatrix=extractMealFeatures(mealData)
mealMatrix_1=extractMealFeatures(mealData_1)
mealMatrix=pd.concat([mealMatrix,mealMatrix_1]).reset_index().drop(columns='index')



# FEATURE EXTRACTION FROM NON-MEAL DATA (SIMILAR TO MEAL DATA):


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

nonMealMatrix=extractNoMealFeatures(noMealData)
nonMealMatrix_1=extractNoMealFeatures(noMealData_1)
nonMealMatrix=pd.concat([nonMealMatrix,nonMealMatrix_1]).reset_index().drop(columns='index')

# USING DECISION TREE CLASSIFIER TO PREDICT 0/1 ALONG WITH K-FOLD CROSS VALIDATION:

mealMatrix['label']=1
nonMealMatrix['label']=0
totData=pd.concat([mealMatrix,nonMealMatrix]).reset_index().drop(columns='index')
dataset=shuffle(totData,random_state=1).reset_index().drop(columns='index')
kFold = KFold(n_splits=10,shuffle=True,random_state=1)
principalData=dataset.drop(columns='label')
scores = []
dTree=DecisionTreeClassifier(criterion="entropy")
for train_index, test_index in kFold.split(principalData):
    x_train,x_test,y_train,y_test = principalData.loc[train_index],principalData.loc[test_index],dataset.label.loc[train_index],dataset.label.loc[test_index]
    dTree.fit(x_train,y_train)
    scores.append(dTree.score(x_test,y_test))


# PRINTING FINAL SCORE:


print('Prediction score with Decision Tree is',np.mean(scores)*100,'%')


model=DecisionTreeClassifier(criterion='entropy')
x, y= principalData, dataset['label']
model.fit(x,y)
dump(model, 'DecisionTreeClassifier.pickle')