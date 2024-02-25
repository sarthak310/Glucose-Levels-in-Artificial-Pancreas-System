import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cluster, silhouette_score, v_measure_score, adjusted_rand_score, completeness_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import warnings
from math import log, e, ceil
from scipy.stats import entropy
from prettytable import PrettyTable

pd.options.display.float_format = '{:20,.4f}'.format

warnings.filterwarnings('ignore')
# %precision 2
# %matplotlib inline
sns.set(font_scale=1)

"""<br><hr/>

## 1) Loading data:
"""

insulinData = pd.read_csv("InsulinData.csv", encoding='latin', index_col=False)

"""<br><hr/>

## 2) Preprocessing:
"""

insulinData = insulinData[["Date","Time","BWZ Carb Input (grams)"]]

insulinData = insulinData.dropna()

scal = StandardScaler()
insulinData["BWZ Carb Input (Scaled)"] = scal.fit_transform(insulinData['BWZ Carb Input (grams)'].values.reshape(-1, 1))

"""<br><hr/>

## 3) Extracting Ground Truth:

> 1. Derive the max and min value of meal intake amount from the Y column of the Insulin data. <br/>
> 2. Discretize the meal amount in bins of size 20. <br/>
> 3. According to their meal amount put them in the respective bins. <br/>

In total, you should have n = (max-min)/20  bins.
"""

# 1.

maxMealAmt = max(insulinData['BWZ Carb Input (grams)'])
minMealAmt = min(insulinData['BWZ Carb Input (grams)'])

print("Max value of meal intake amount in grams:", maxMealAmt)
print("Min value of meal intake amount in grams:", minMealAmt)

numOfBins = ceil((maxMealAmt-minMealAmt)/20)

print("n = (",maxMealAmt,"-",minMealAmt,") / 20 =", numOfBins, "bins")

# 2.

dictOfBins = {
    1:[0,20],
    2:[21,40],
    3:[41,60],
    4:[61,80],
    5:[81,100],
    6:[101,120],
    7:[121,140]
}

def getBins(mealAmt):
    lint = 0
    for num, range in dictOfBins.items():
        if range[0] <= mealAmt <= range[1]:
            lint = num
    return lint

# 3.

insulinData['Ground Truth'] = insulinData['BWZ Carb Input (grams)'].apply(lambda x: getBins(x))

"""<br><hr/>

## 4) Clustering:

> 1. Feature Selection <br/>
> 2. Methods to calculate accuracy based on SSE, entropy and purity metrics
> 2. KMeans Clustering <br/>
> 3. DBSCAN Clustering <br/>
> 4. Accuracy Report
"""

# 1.

x = insulinData["BWZ Carb Input (Scaled)"].values.reshape(-1,1)

# 2.

def calcEntropy(y_true, y_pred, base = 2):
    ConfusionMatrix = cluster.contingency_matrix(y_true, y_pred)
    base = e if base is None else base
    
    Entropy = []

    for i in range(0, len(ConfusionMatrix)):
        p = ConfusionMatrix[i,:]
        p = pd.Series(p).value_counts(normalize=True, sort=False)
        Entropy.append((-p/p.sum() * np.log(p/p.sum())/np.log(2)).sum())
    
    pTotal = sum(ConfusionMatrix,1);
    wholeEntropy = 0;

    for i in range(0, len(ConfusionMatrix)):
        p = ConfusionMatrix[i,:]
        wholeEntropy = wholeEntropy + ((sum(p))/(sum(pTotal)))*Entropy[i]
    
    return wholeEntropy

def calcPurityScore(y_true, y_pred):
    ConfusionMatrix = cluster.contingency_matrix(y_true, y_pred)

    Purity = []

    for i in range(0, len(ConfusionMatrix)):
        p = ConfusionMatrix[i,:]
        Purity.append(p.max()/p.sum())

    pTotal = sum(ConfusionMatrix,1);
    WholePurity = 0;

    for i in range(0, len(ConfusionMatrix)):
        p = ConfusionMatrix[i,:]
        WholePurity = WholePurity + ((sum(p))/(sum(pTotal)))*Purity[i]
    
    return WholePurity

def calcMeasureScore(y_true, y_pred):
    return v_measure_score(y_true, y_pred)

# 3.

kmeans = KMeans(n_clusters=7, random_state=42, max_iter=100)
kmeansModel = kmeans.fit(x)

print("SSE:"+ str(kmeans.inertia_))

kmeansSilhouette = silhouette_score(x, kmeans.labels_).round(2)
print("\nSilhouette Score:"+str(kmeansSilhouette))

insulinData['KmeanCluster'] = kmeansModel.predict(x)

kmeansEntropy = calcEntropy(insulinData['Ground Truth'],insulinData['KmeanCluster'])
kmeansPurityScore = calcPurityScore(insulinData['Ground Truth'],insulinData['KmeanCluster'])
kmeansVMeasureScore = calcMeasureScore(insulinData['Ground Truth'], insulinData['KmeanCluster'])

# 4.

dbscan = DBSCAN(eps=0.3)
dbscanModel = dbscan.fit(x)

dbscanSilhouette = silhouette_score(x, dbscanModel.labels_).round(2)
print("Silhouette Score:"+str(dbscanSilhouette))

insulinData['DBSCAN_Cluster'] = dbscanModel.fit_predict(x)

dbscanEntropy = calcEntropy(insulinData['Ground Truth'],insulinData['DBSCAN_Cluster'])
dbscanPurityScore = calcPurityScore(insulinData['Ground Truth'],insulinData['DBSCAN_Cluster'])
dbscanVMeasureScore = calcMeasureScore(insulinData['Ground Truth'], insulinData['DBSCAN_Cluster'])

# 5.

modelAccuracy = PrettyTable()
modelAccuracy.field_names = ["","SSE","V-Measure Score", "Entropy", "Purity Metrics"]
modelAccuracy.align[""] = "r"
modelAccuracy.add_row(["K-Means","%.2f"%kmeans.inertia_,"%.2f"%kmeansVMeasureScore,"%.2f"%kmeansEntropy,"%.2f"%kmeansPurityScore])
modelAccuracy.add_row(["DBSCAN","-","%.2f"%dbscanVMeasureScore,"%.2f"%dbscanEntropy,"%.2f"%dbscanPurityScore])
print("Accuracy Report:\n")
print(modelAccuracy)

results = {}

results['SSE for Kmeans'] =  "%.2f"%kmeans.inertia_
results['SSE for DBSCAN'] =  "-"
results['Entropy for KMeans'] =  "%.2f"%kmeansEntropy
results['Entropy for DBSCAN'] =  "%.2f"%dbscanEntropy
results['Purity for KMeans'] =  "%.2f"%kmeansPurityScore
results['Purity for DBSCAN'] =  "%.2f"%dbscanPurityScore

resultsDf = pd.DataFrame(results, index=[0])
resultsDf.to_csv('Result.csv',index=False)