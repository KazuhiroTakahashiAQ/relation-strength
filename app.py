import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as mpl
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from data_processing import load_source_data
from plot2d import create_2d_figure
from plot3d import create_3d_figure
from plot_network import create_network_graph

# Macの標準日本語フォント「Hiragino Sans」を指定
mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False

st.title("Strength Finder 可視化アプリ")

# サイドバーに表示モードの選択とクラスタ数の設定を移動
st.sidebar.subheader("表示モードの選択")
view_mode = st.sidebar.selectbox(
    "表示モード", options=["2D Scatter Plot", "3D Scatter Plot", "ネットワークグラフ"]
)

st.sidebar.subheader("クラスタ数の設定")
n_clusters = st.sidebar.slider("クラスタ数", min_value=2, max_value=10, value=10)

# データ読み込み
df_source, rows = load_source_data("data.txt")

# （元データの表は描画しない。AgGrid表で置き換え）
st.subheader("強調表示オプション（元データの行選択）")

# AgGrid用に行番号を追加
df_table = df_source.copy()
df_table.insert(0, "Row", ["行{}".format(i + 1) for i in range(len(df_source))])

# AgGrid設定
grid_options = GridOptionsBuilder.from_dataframe(df_table)
grid_options.configure_selection(selection_mode="multiple", use_checkbox=True)
grid_opts = grid_options.build()

# AgGrid表示
aggrid_response = AgGrid(
    df_table,
    gridOptions=grid_opts,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=300,
    width="100%",
)
selected_rows = aggrid_response["selected_rows"]

# 選択された行のインデックスを抽出
highlight_indices = []
for row in selected_rows:
    row_str = row.get("Row", "")
    if row_str.startswith("行"):
        try:
            idx = int(row_str.replace("行", "")) - 1
            highlight_indices.append(idx)
        except ValueError:
            pass
highlight_indices = list(set(highlight_indices))

# ワンホットベクトル化
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(rows)

# 表示モードに応じて可視化
if view_mode == "2D Scatter Plot":
    # PCA 2次元
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(binary_matrix)

    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    # 可視化用DF
    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters.astype(str)

    # Plotly 2D散布図（ホバー＋強調表示）
    fig = create_2d_figure(df_reduced, highlight_indices=highlight_indices)
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "3D Scatter Plot":
    # PCA 3次元
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(binary_matrix)

    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    # 可視化用DF
    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters.astype(str)

    # Plotly 3D散布図（ホバー＋強調表示）
    fig = create_3d_figure(df_reduced, highlight_indices=highlight_indices)
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "ネットワークグラフ":
    # ネットワークグラフ（行選択の強調はせず、共起関係を表示）
    fig = create_network_graph(rows)
    st.plotly_chart(fig, use_container_width=True)
