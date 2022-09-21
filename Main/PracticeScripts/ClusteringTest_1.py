# Agglomerative clustering of the univariate dataset is tested here
# Electricity dataset with 321 columsn
from data_provider.data_factory import data_provider
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

dataPath = '..\..\Datasets\electricity\electricity.csv'
df_raw = pd.read_csv(dataPath)
scaler = StandardScaler()

cols = list(df_raw.columns)
cols.remove('date')
cols.remove('OT')
Data = df_raw[cols] # all data except ....
Data_scaled = scaler.fit_transform(Data)
# Clustering algo initialization
aggloclust=AgglomerativeClustering(n_clusters=3).fit(Data_scaled.transpose())
print(aggloclust)
labels = aggloclust.labels_
plt.figure(); plt.hist(labels)

# Separating into diff dataframes
cl_0 = np.array(Data)[:,labels==0]
cl_0m = np.mean(cl_0, axis=1)
cl_1 = np.array(Data)[:,labels==1]
cl_1m = np.mean(cl_1, axis=1)
cl_2 = np.array(Data)[:,labels==2]
cl_2m = np.mean(cl_2, axis=1)

plt.figure(); plt.plot(cl_0m[:250]); plt.plot(cl_1m[:250]); plt.plot(cl_2m[:250])

# creating dataframes for csv saving
cl_0 = pd.DataFrame(cl_0, columns=range(cl_0.shape[1]))
cl_0.insert(0, 'date', df_raw['date'])
cl_1 = pd.DataFrame(cl_1, columns=range(cl_1.shape[1]))
cl_1.insert(0, 'date', df_raw['date'])
cl_2 = pd.DataFrame(cl_2, columns=range(cl_2.shape[1]))
cl_2.insert(0, 'date', df_raw['date'])

cl_0.to_csv('electricity_cl_0.csv', index=False)
cl_1.to_csv('electricity_cl_1.csv', index=False)
cl_2.to_csv('electricity_cl_2.csv', index=False)
# Saved for future usage