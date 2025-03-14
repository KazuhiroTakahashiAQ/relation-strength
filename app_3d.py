import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from data_processing import load_source_data

st.title("Strength Finder インタラクティブ 3D 散布図アプリ")

# データの読み込み
df, rows = load_source_data("data.txt")
st.subheader("元データ")
st.write(df)

# 各行の強みをワンホットベクトルに変換
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(rows)

# 次元削減：PCAで3次元に変換
pca = PCA(n_components=3)
reduced = pca.fit_transform(binary_matrix)

# サイドバー：クラスタ数の設定（デフォルト値は10）
st.sidebar.subheader("クラスタ数の設定")
n_clusters = st.sidebar.slider("クラスタ数", min_value=2, max_value=10, value=10)

# KMeansクラスタリングの実施（3次元PCAデータに対して）
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced)

# PCA の結果とクラスタ情報を DataFrame に格納
df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
df_reduced["Cluster"] = clusters.astype(str)  # Plotly で色分けするため文字列に変換

# Plotly Express を使ってインタラクティブな3D散布図を作成
fig = px.scatter_3d(
    df_reduced,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Cluster",
    hover_data=["Data"],
    title="Strength Finder 3D インタラクティブ散布図",
    width=800,
    height=600,
)

st.plotly_chart(fig, use_container_width=True)
