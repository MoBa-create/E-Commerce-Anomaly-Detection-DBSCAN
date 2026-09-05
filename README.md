# E-Commerce Anomaly & Outlier Detection using DBSCAN

An Unsupervised Machine Learning project designed to detect potential fraudulent transactions and behavioral outliers in E-commerce data using **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).

## 📌 Project Overview
Unlike traditional distance-based algorithms like K-Means, **DBSCAN** clusters data based on spatial density without requiring a predefined number of clusters ($K$). Crucially, it identifies sparse, isolated points as **Noise (-1)**, making it ideal for fraud detection, quality control, and cybersecurity anomaly detection.

![DBSCAN Clustering Result](outputs/dbscan_clusters.png)

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib

## 📊 Key Findings
* **Automated Clustering:** Identified dense behavioral clusters without specifying $K$.
* **Outlier Isolation:** Successfully flagged isolated high-risk transactions (Noise / Outliers) based on transaction frequency and amount.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MoBa-create/E-Commerce-Anomaly-Detection-DBSCAN.git


1 : Install dependencies:
	pip install -r requirements.txt

2 : Execute the script:
	python dbscan_anomaly_detection.py
