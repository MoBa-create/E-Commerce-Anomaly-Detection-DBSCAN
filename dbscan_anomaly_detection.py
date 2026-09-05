from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

np.random.seed(42)

normal_transactions_1 = np.random.normal(loc=[50, 10], scale=[10, 2], size=(250, 2))
normal_transactions_2 = np.random.normal(loc=[200, 3], scale=[25, 1], size=(150, 2))

anomalies = np.array([
    [500, 25],
    [10, 30],
    [450, 2],
    [20, 28],
    [600, 1]
])

data = np.vstack([normal_transactions_1, normal_transactions_2, anomalies])
df = pd.DataFrame(data, columns=['Transaction_Amount', 'Transaction_Frequency'])

print(df.head())
print(f"\n Total number of transactions : {len(df)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Transaction_Amount', 'Transaction_Frequency']])

dbscan = DBSCAN(eps=0.35, min_samples=6)
df['Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
n_noise = list(df['Cluster']).count(-1)

print("\n=== Analysis results DBSCAN ===")
print(f"Number of natural behavioral groups discovered : {n_clusters}")
print(f"Number of abnormal/fraudulent transactions detected(Outliers) : {n_noise}")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x='Transaction_Amount', 
    y='Transaction_Frequency', 
    hue='Cluster', 
    palette='Set1', 
    data=df, 
    s=80,
    style=(df['Cluster'] == -1)
)

plt.title(f'E-Commerce Anomaly Detection using DBSCAN\nDetected Outliers (Noise): {n_noise}')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Transaction Frequency (Per Month)')
plt.grid(True)

plt.savefig(os.path.join(OUTPUTS_DIR, 'dbscan_clusters.png'))
plt.show()

joblib.dump(scaler, os.path.join(OUTPUTS_DIR, 'scaler.pkl'))
joblib.dump(dbscan, os.path.join(OUTPUTS_DIR, 'dbscan_model.pkl'))

print(f"\n The images and templates were successfully saved inside the folder: {OUTPUTS_DIR}")