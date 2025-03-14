import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as mpl

from data_processing import load_source_data
from plot2d import create_2d_figure, get_legend_df
from plot3d import create_3d_figure

# Macの標準日本語フォント「Hiragino Sans」を指定
mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False

st.title("Strength Finder 可視化アプリ")

# 表示モードの切り替え
view_mode = st.radio("表示モードを選択", ("2D", "3D"))

# サイドバー：クラスタ数の設定（デフォルト値は10）
st.sidebar.subheader("クラスタ数の設定")
n_clusters = st.sidebar.slider("クラスタ数", min_value=2, max_value=10, value=10)

# ソースデータの読み込み
df_source, rows = load_source_data("data.txt")
st.subheader("元データ")
st.write(df_source)

# ユーザーが元データの行を選択して強調表示できるようにする
options = ["行{}".format(i + 1) for i in range(len(rows))]
selected_rows = st.multiselect("元データの行を選択して強調表示", options=options)
# 選択された行のインデックスを取得
highlight_indices = (
    [int(x.replace("行", "")) - 1 for x in selected_rows] if selected_rows else []
)

# 各行の強みをワンホットベクトルに変換
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(rows)

if view_mode == "2D":
    # 2次元PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(binary_matrix)
    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    # DataFrame作成
    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters
    # 2D散布図作成（highlight_indices を渡す）
    fig, scatter = create_2d_figure(df_reduced, highlight_indices=highlight_indices)
    legend_df = get_legend_df(df_reduced)
    # 2カラム表示：散布図と凡例
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.subheader("Legend")
        st.table(legend_df)
else:
    # 3次元PCA
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(binary_matrix)
    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    # DataFrame作成
    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters.astype(str)  # Plotlyでは文字列で扱う
    # 3D散布図作成（highlight_indices を渡す）
    fig = create_3d_figure(df_reduced, highlight_indices=highlight_indices)
    st.plotly_chart(fig, use_container_width=True)
