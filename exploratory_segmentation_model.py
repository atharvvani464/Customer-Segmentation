import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Step 1: Load dataset
df = pd.read_csv("customer_segmentation.csv")

# Step 2: Feature Engineering
df["Total_Spending"] = df[[
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)

df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Customer_Since_Days"] = (pd.to_datetime("2025-01-01") - pd.to_datetime(df["Dt_Customer"], dayfirst=True)).dt.days

# Step 3: Select and Scale Features
features = ["Income", "Recency", "NumWebPurchases", "NumStorePurchases",
            "NumWebVisitsMonth", "Total_Spending", "Children", "Customer_Since_Days"]
X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Step 4: Determine Optimal k using Silhouette Score
silhouette_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Optimal k using Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Train Final KMeans Model
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)
joblib.dump(kmeans, "kmeans_model.pkl")

# Step 6: Visualize Clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Cluster"] = cluster_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2")
plt.title("Customer Segments Visualized with PCA")
plt.tight_layout()
plt.show()

# Step 7: Profile the Clusters
X["Cluster"] = cluster_labels
cluster_summary = X.groupby("Cluster").mean().round(2)
print("\nCluster Profiles:\n")
print(cluster_summary)
