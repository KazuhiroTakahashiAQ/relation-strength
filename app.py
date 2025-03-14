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

mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False

# セッションステートで「再描画済み」かどうかを管理
if "already_rerun" not in st.session_state:
    st.session_state["already_rerun"] = False

st.title("ベクトル 可視化アプリ")

# --- サイドバー ---
st.sidebar.subheader("表示モードの選択")
view_mode = st.sidebar.selectbox(
    "表示モード", options=["3D Scatter Plot", "2D Scatter Plot", "ネットワークグラフ"]
)

st.sidebar.subheader("クラスタ数の設定")
n_clusters = st.sidebar.slider("クラスタ数", min_value=2, max_value=10, value=10)

# データ読み込み
with st.spinner("データを読み込んでいます..."):
    df_source, rows = load_source_data("data.txt")

st.subheader("強調表示オプション（元データの行選択）")

df_table = df_source.copy()
df_table.insert(0, "Row", ["行{}".format(i + 1) for i in range(len(df_source))])

# AgGrid設定
grid_options = GridOptionsBuilder.from_dataframe(df_table)
grid_options.configure_selection(selection_mode="multiple", use_checkbox=True)
grid_options.configure_grid_options(domLayout="normal")
grid_opts = grid_options.build()

aggrid_response = AgGrid(
    df_table,
    gridOptions=grid_opts,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=True,
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

# ワンホットベクトル化
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(rows)

# ---- 表示モードに応じた可視化 ----
if view_mode == "2D Scatter Plot":
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(binary_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
    df_reduced["Cluster"] = clusters.astype(str)

    fig = create_2d_figure(df_reduced, highlight_indices=highlight_indices)
    st.plotly_chart(fig, use_container_width=True)

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
    fig = create_network_graph(rows)
    st.plotly_chart(fig, use_container_width=True)

# ---- ここで初回だけ再描画する ----
if not st.session_state["already_rerun"]:
    # 初回のみ再描画してフラグを更新
    st.session_state["already_rerun"] = True
    st.experimental_rerun()
