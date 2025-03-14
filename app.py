import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as mpl
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from data_processing import load_source_data
from plot2d import create_2d_figure, get_legend_df
from plot3d import create_3d_figure
from plot_network import create_network_graph

# Macの標準日本語フォント「Hiragino Sans」を指定
mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False

st.title("Strength Finder 可視化アプリ")

# 1. 表示モード選択（ドロップダウン）
view_mode = st.selectbox(
    "表示モードを選択",
    options=["2D Scatter Plot", "3D Scatter Plot", "ネットワークグラフ"],
)

# サイドバー：クラスタ数の設定（デフォルト値は10）
st.sidebar.subheader("クラスタ数の設定")
n_clusters = st.sidebar.slider("クラスタ数", min_value=2, max_value=10, value=10)

# ソースデータの読み込み
df_source, rows = load_source_data("data.txt")

# AgGrid 用に行番号を追加したデータフレームを作成
df_table = df_source.copy()
df_table.insert(0, "Row", ["行{}".format(i + 1) for i in range(len(df_source))])

# 2. 強調表示オプション（元データの行選択）を AgGrid で実装
st.subheader("強調表示オプション（元データの行選択）")
grid_options = GridOptionsBuilder.from_dataframe(df_table)
grid_options.configure_selection(selection_mode="multiple", use_checkbox=True)
grid_options.configure_grid_options(domLayout="normal")
grid_opts = grid_options.build()
aggrid_response = AgGrid(
    df_table,
    gridOptions=grid_opts,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=300,
    width="100%",
)
selected_rows = aggrid_response["selected_rows"]
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

# 各行の強みをワンホットベクトルに変換
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(rows)

# 3. 散布図／ネットワークグラフの表示
if view_mode == "2D Scatter Plot":
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(binary_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters
    fig, _ = create_2d_figure(df_reduced, highlight_indices=highlight_indices)
    legend_df = get_legend_df(df_reduced)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.subheader("Legend")
        st.table(legend_df)
elif view_mode == "3D Scatter Plot":
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(binary_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters.astype(str)
    fig = create_3d_figure(df_reduced, highlight_indices=highlight_indices)
    st.plotly_chart(fig, use_container_width=True)
elif view_mode == "ネットワークグラフ":
    # ネットワークグラフは、各強みの共起関係を表示（行選択の強調は行いません）
    fig = create_network_graph(rows)
    st.plotly_chart(fig, use_container_width=True)

# 4. 元データの表示（散布図の下に表示）
st.subheader("元データ")
st.write(df_source)
