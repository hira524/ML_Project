import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import silhouette_score  # You missed this import

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# === Dataset Paths ===
DATA_PATH_1 = r"C:\Users\Syeda Hira\OneDrive\Documents\hi\ML_Project\D1\D1_adult_cencus_income_data.csv"
DATA_PATH_2 = r"C:\Users\Syeda Hira\OneDrive\Documents\hi\ML_Project\S2\ai4i2020.csv"
DATA_PATH_3 = r"C:\Users\Syeda Hira\OneDrive\Documents\hi\ML_Project\D3\AmesHousing.csv"


# === Utility Functions ===
def print_metrics(y_true, y_pred, name="Model"):
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score(y_true, y_pred), 4))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def evaluate_model_with_heatmap(model, X_test, y_test, labels, title=""):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print_metrics(y_test, y_pred, title)


# === Dataset 1: Adult Census ===
def process_adult_dataset():
    print(f"Loading dataset from: {DATA_PATH_1}")
    df = pd.read_csv(DATA_PATH_1)

    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    X = df.drop('income', axis=1)
    y = df['income']

    # Feature Selection (Chi2)
    X_scaled = MinMaxScaler().fit_transform(X)
    selector = SelectKBest(score_func=chi2, k=10)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]
    X = pd.DataFrame(X_selected, columns=selected_features)

    return X, y


def clustering_validation(X):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    print("KMeans Silhouette Score:", silhouette_score(X, kmeans_labels))

    dbscan = DBSCAN(eps=2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    print("DBSCAN Silhouette Score:", silhouette_score(X, dbscan_labels))


def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)

    print("Original:", Counter(y))
    print("SMOTE:", Counter(y_sm))
    print("Under:", Counter(y_rus))

    return X_sm, y_sm, X_rus, y_rus


def train_classifiers(X, y, name=""):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print_metrics(y_test, y_pred, f"{model_name} ({name})")


# === Dataset 2: Predictive Maintenance ===
def process_predictive_maintenance():
    print(f"Loading dataset from: {DATA_PATH_2}")
    df = pd.read_csv(DATA_PATH_2)
    df = df.drop(columns=[col for col in ['UDI', 'Product ID'] if col in df.columns])

    X = pd.get_dummies(df.drop(columns=['Machine failure']), drop_first=True)
    y = df['Machine failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def feature_extraction_pca(X_train, X_test):
    pca = PCA(n_components=10)
    return pca.fit_transform(X_train), pca.transform(X_test)


def feature_extraction_autoencoder(X_train, X_test):
    input_dim = X_train.shape[1]
    encoding_dim = 10

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded_output = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded_output = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(input_layer, decoded_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32,
                    shuffle=True, validation_data=(X_test, X_test), verbose=0)

    encoder = Model(inputs=input_layer, outputs=encoded_output)
    return encoder.predict(X_train), encoder.predict(X_test)


# === Dataset 3: Ames Housing ===
def process_ames_dataset():
    print(f"Loading dataset from: {DATA_PATH_3}")
    df = pd.read_csv(DATA_PATH_3)
    df = df.dropna(subset=['SalePrice'])

    df['PriceCategory'] = pd.qcut(df['SalePrice'], q=3, labels=["Low", "Medium", "High"])
    y = df['PriceCategory']

    df = df.drop(columns=['SalePrice', 'Order', 'PriceCategory'], errors='ignore')
    X = df.select_dtypes(include=[np.number]).dropna(axis=1)

    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# === Main ===
if __name__ == "__main__":
    # Adult Dataset
    X_adult, y_adult = process_adult_dataset()
    clustering_validation(X_adult)
    X_sm, y_sm, X_rus, y_rus = balance_data(X_adult, y_adult)
    train_classifiers(X_sm, y_sm, "SMOTE")
    train_classifiers(X_rus, y_rus, "Undersampling")

    # Predictive Maintenance
    X_train_pm, X_test_pm, y_train_pm, y_test_pm = process_predictive_maintenance()
    X_train_pca, X_test_pca = feature_extraction_pca(X_train_pm, X_test_pm)
    clf_pca = RandomForestClassifier().fit(X_train_pca, y_train_pm)
    evaluate_model_with_heatmap(clf_pca, X_test_pca, y_test_pm, [0, 1], "PCA")

    X_train_ae, X_test_ae = feature_extraction_autoencoder(X_train_pm, X_test_pm)
    clf_ae = RandomForestClassifier().fit(X_train_ae, y_train_pm)
    evaluate_model_with_heatmap(clf_ae, X_test_ae, y_test_pm, [0, 1], "Autoencoder")

    # Ames Housing
    X_train_ames, X_test_ames, y_train_ames, y_test_ames = process_ames_dataset()

    clf_orig = RandomForestClassifier().fit(X_train_ames, y_train_ames)
    evaluate_model_with_heatmap(clf_orig, X_test_ames, y_test_ames, ["Low", "Medium", "High"], "Original")

    smote = SMOTE(random_state=42)
    X_res_sm, y_res_sm = smote.fit_resample(X_train_ames, y_train_ames)
    clf_sm = RandomForestClassifier().fit(X_res_sm, y_res_sm)
    evaluate_model_with_heatmap(clf_sm, X_test_ames, y_test_ames, ["Low", "Medium", "High"], "SMOTE")

    rus = RandomUnderSampler(random_state=42)
    X_res_rus, y_res_rus = rus.fit_resample(X_train_ames, y_train_ames)
    clf_rus = RandomForestClassifier().fit(X_res_rus, y_res_rus)
    evaluate_model_with_heatmap(clf_rus, X_test_ames, y_test_ames, ["Low", "Medium", "High"], "Undersampling")
