import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans


df=pd.read_csv("insurance.csv")
print(df.head())
df=df.drop_duplicates()
df=df.dropna()
print(df)
x=df.select_dtypes(include=np.number)
print(x)
scaler =StandardScaler()
x=scaler.fit_transform(x)
print(x)
scr=[]
kvalues= range(2,21)

for k in kvalues:
    model=KMedoids(n_clusters=k)
    labels=model.fit_predict(x)
    score=silhouette_score(x,labels)
    scr.append(score)
plt.plot(kvalues,scr)
plt.xlabel("k")
plt.ylabel("score")
plt.show()
bestK=kvalues[np.argmax(scr)]
print("best k",bestK)
inertia = []

for k in kvalues:
    model=KMeans(n_clusters=k, random_state=42)
    model.fit(x)
    inertia.append(model.inertia_)
plt.subplot(1,2,2)
plt.plot(kvalues, inertia, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()    
model=KMedoids(n_clusters=bestK,random_state=42)
labels=model.fit_predict(x)
df["cluster"]=labels
print(df.groupby("cluster").mean(numeric_only=True))
print(df["cluster"].value_counts())
print(df)
