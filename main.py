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


from sklearn.ensemble import ExtraTreesClassifier

X = df_scaled.drop(columns=['target'])  # Replace 'target' with your actual target column
y = df_scaled['target']

model = ExtraTreesClassifier()
model.fit(X, y)
importances = model.feature_importances_

selected_features = X.columns[importances > 0.05]  # threshold can vary
X_selected = X[selected_features]


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Apply SMOTE
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_selected, y)

# Apply Random Under Sampling
under = RandomUnderSampler(random_state=42)
X_under, y_under = under.fit_resample(X_selected, y)



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
