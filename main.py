import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv(r'C:\Users\Syeda Hira\Downloads\D1\train.csv')
print("Initial Data:\n", df.head())

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_data, columns=df.columns)


from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(df_scaled)
print("Silhouette Score (KMeans):", silhouette_score(df_scaled, kmeans_labels))

# DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_scaled)
print("Silhouette Score (DBSCAN):", silhouette_score(df_scaled, dbscan_labels))
