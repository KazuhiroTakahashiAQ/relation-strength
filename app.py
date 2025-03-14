import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import matplotlib as mpl

from data_processing import get_processed_data

# Macの標準日本語フォント「Hiragino Sans」を指定
mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False  # マイナス記号の対策

st.title("Strength Finder 散布図可視化アプリ（色分け付き）")

# データの読み込みと整形
df, rows, binary_matrix, reduced = get_processed_data("data.txt")

st.subheader("元データ")
st.write(df)

# サイドバー：クラスタ数の設定（デフォルト値は10）
st.sidebar.subheader("クラスタ数の設定")
n_clusters = st.sidebar.slider("クラスタ数", min_value=2, max_value=10, value=10)

# PCA後の2次元データに対してKMeansクラスタリングを実施
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced)

# PCAの結果とクラスタ情報をDataFrameに格納
df_reduced = pd.DataFrame(reduced, columns=["PC1", "PC2"])
df_reduced["Data"] = ["行{}".format(i + 1) for i in range(len(rows))]
df_reduced["Cluster"] = clusters

# 散布図の作成
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df_reduced["PC1"], df_reduced["PC2"], c=df_reduced["Cluster"], cmap="tab10", s=50
)
for i, txt in enumerate(df_reduced["Data"]):
    ax.annotate(
        txt,
        (df_reduced["PC1"][i], df_reduced["PC2"][i]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Strength Finder データの散布図")


# カスタム凡例作成：クラスタ番号と対応する色（tab10カラーマップ）
def rgba_to_hex(rgba):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


legend_items = []
unique_clusters = sorted(set(clusters))
for i in unique_clusters:
    rgba = cm.tab10(i)
    hex_color = rgba_to_hex(rgba)
    legend_items.append({"Cluster": i, "Color": hex_color})
legend_df = pd.DataFrame(legend_items)

# st.columns()を利用して散布図と凡例を横並びに表示
col1, col2 = st.columns([3, 1])
with col1:
    st.pyplot(fig)
with col2:
    st.subheader("Legend")
    st.table(legend_df)
