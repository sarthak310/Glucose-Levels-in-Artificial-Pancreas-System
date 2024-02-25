import pandas as pd
import numpy as np

dataCGM=pd.read_csv('CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
dataInsulin=pd.read_csv('InsulinData.csv',low_memory=False)

#String date time -> date time object
dataCGM['stampDateTime']=pd.to_datetime(dataCGM['Date'] + ' ' + dataCGM['Time'])

#For missing values
discard_dates=dataCGM[dataCGM['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()
dataCGM=dataCGM.set_index('Date').drop(index=discard_dates).reset_index()

testCGM=dataCGM.copy()
testCGM=testCGM.set_index(pd.DatetimeIndex(dataCGM['stampDateTime']))

dataInsulin['stampDateTime']=pd.to_datetime(dataInsulin['Date'] + ' ' + dataInsulin['Time'])

#Auto mode starts when we get AUTO MODE ACTIVE PLGM OFF message in InsulinData
startAM=dataInsulin.sort_values(by='stampDateTime',ascending=True).loc[dataInsulin['Alarm']=='AUTO MODE ACTIVE PLGM OFF'].iloc[0]['stampDateTime']
#Time stamp in dataCGM nearest to (later than) auto mode start in InsulinData
dataframeAM=dataCGM.sort_values(by='stampDateTime',ascending=True).loc[dataCGM['stampDateTime']>=startAM]
#Before auto mode is manual mode
dataframeMM=dataCGM.sort_values(by='stampDateTime',ascending=True).loc[dataCGM['stampDateTime']<startAM]


#..........................AUTO MODE (AM).............................


dateIndexAM=dataframeAM.copy()
dateIndexAM=dateIndexAM.set_index('stampDateTime')

#dropping NaN values
listAM=dateIndexAM.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
dateIndexAM=dateIndexAM.loc[dateIndexAM['Date'].isin(listAM)]

#Percentage time in hyperglycemia (CGM > 180 mg/dL) -> whole day (WD), daytime (DT), overnight (ON)

normalWD_AM=(dateIndexAM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
normalDT_AM=(dateIndexAM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
normalON_AM=(dateIndexAM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage of time in hyperglycemia critical (CGM > 250 mg/dL) -> whole day, daytime, overnight

criticalWD_AM=(dateIndexAM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
criticalDT_AM=(dateIndexAM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
criticalON_AM=(dateIndexAM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) -> whole day, daytime, overnight

rangeWD_AM=(dateIndexAM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexAM['Sensor Glucose (mg/dL)']>=70) & (dateIndexAM['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangeDT_AM=(dateIndexAM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexAM['Sensor Glucose (mg/dL)']>=70) & (dateIndexAM['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangeON_AM=(dateIndexAM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexAM['Sensor Glucose (mg/dL)']>=70) & (dateIndexAM['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) -> whole day, daytime, overnight

rangesecWD_AM=(dateIndexAM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexAM['Sensor Glucose (mg/dL)']>=70) & (dateIndexAM['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangesecDT_AM=(dateIndexAM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexAM['Sensor Glucose (mg/dL)']>=70) & (dateIndexAM['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangesecON_AM=(dateIndexAM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexAM['Sensor Glucose (mg/dL)']>=70) & (dateIndexAM['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in hypoglycemia level 1 (CGM < 70 mg/dL) -> whole day, daytime, overnight

level1WD_AM=(dateIndexAM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level1DT_AM=(dateIndexAM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level1ON_AM=(dateIndexAM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in hypoglycemia level 2 (CGM < 54 mg/dL) -> whole day, daytime, overnight

level2WD_AM=(dateIndexAM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level2DT_AM=(dateIndexAM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level2ON_AM=(dateIndexAM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexAM['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


#.............................MANUAL MODE (MM)....................................


dateIndexMM=dataframeMM.copy()
dateIndexMM=dateIndexMM.set_index('stampDateTime')

listMM=dateIndexMM.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>0.8*288).dropna().index.tolist()
dateIndexMM=dateIndexMM.loc[dateIndexMM['Date'].isin(listMM)]

#Percentage time in hyperglycemia (CGM > 180 mg/dL) -> whole day, daytime, overnight

normalWD_MM=(dateIndexMM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
normalDT_MM=(dateIndexMM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
normalON_MM=(dateIndexMM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage of time in hyperglycemia critical (CGM > 250 mg/dL) -> whole day, daytime, overnight

criticalWD_MM=(dateIndexMM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
criticalDT_MM=(dateIndexMM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
criticalON_MM=(dateIndexMM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) -> whole day, daytime, overnight

rangeWD_MM=(dateIndexMM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexMM['Sensor Glucose (mg/dL)']>=70) & (dateIndexMM['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangeDT_MM=(dateIndexMM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexMM['Sensor Glucose (mg/dL)']>=70) & (dateIndexMM['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangeON_MM=(dateIndexMM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexMM['Sensor Glucose (mg/dL)']>=70) & (dateIndexMM['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) -> whole day, daytime, overnight

rangesecWD_MM=(dateIndexMM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexMM['Sensor Glucose (mg/dL)']>=70) & (dateIndexMM['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangesecDT_MM=(dateIndexMM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexMM['Sensor Glucose (mg/dL)']>=70) & (dateIndexMM['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
rangesecON_MM=(dateIndexMM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[(dateIndexMM['Sensor Glucose (mg/dL)']>=70) & (dateIndexMM['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in hypoglycemia level 1 (CGM < 70 mg/dL) -> whole day, daytime, overnight

level1WD_MM=(dateIndexMM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level1DT_MM=(dateIndexMM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level1ON_MM=(dateIndexMM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#percentage time in hypoglycemia level 2 (CGM < 54 mg/dL) -> whole day, daytime, overnight

level2WD_MM=(dateIndexMM.between_time('0:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level2DT_MM=(dateIndexMM.between_time('6:00:00','23:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)
level2ON_MM=(dateIndexMM.between_time('0:00:00','05:59:59')[['Date','Time','Sensor Glucose (mg/dL)']].loc[dateIndexMM['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)


#.......................................RESULTS.....................................


dataframeResults = pd.DataFrame({
                           'Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)':[ normalON_MM.mean(axis=0),normalON_AM.mean(axis=0)],
                           'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)':[ criticalON_MM.mean(axis=0),criticalON_AM.mean(axis=0)],
                           'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)':[ rangeON_MM.mean(axis=0),rangeON_AM.mean(axis=0)],
                           'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)':[ rangesecON_MM.mean(axis=0),rangesecON_AM.mean(axis=0)],
                           'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)':[ level1ON_MM.mean(axis=0),level1ON_AM.mean(axis=0)],
                           'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)':[ np.nan_to_num(level2ON_MM.mean(axis=0)),level2ON_AM.mean(axis=0)],
                           
                           'Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)':[ normalDT_MM.mean(axis=0),normalDT_AM.mean(axis=0)],
                           'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)':[ criticalDT_MM.mean(axis=0),criticalDT_AM.mean(axis=0)],
                           'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)':[ rangeDT_MM.mean(axis=0),rangeDT_AM.mean(axis=0)],
                           'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)':[ rangesecDT_MM.mean(axis=0),rangesecDT_AM.mean(axis=0)],
                           'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)':[ level1DT_MM.mean(axis=0),level1DT_AM.mean(axis=0)],
                           'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)':[ level2DT_MM.mean(axis=0),level2DT_AM.mean(axis=0)],
                           
                           'Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)':[ normalWD_MM.mean(axis=0),normalWD_AM.mean(axis=0)],
                           'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)':[ criticalWD_MM.mean(axis=0),criticalWD_AM.mean(axis=0)],
                           'Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)':[ rangeWD_MM.mean(axis=0),rangeWD_AM.mean(axis=0)],
                           'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)':[ rangesecWD_MM.mean(axis=0),rangesecWD_AM.mean(axis=0)],
                           'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)':[ level1WD_MM.mean(axis=0),level1WD_AM.mean(axis=0)],
                           'Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)':[ level2WD_MM.mean(axis=0),level2WD_AM.mean(axis=0)]
                                              
},
                            index=['Manual Mode','Auto Mode'])

dataframeResults.to_csv('Results.csv',header=False,index=False)