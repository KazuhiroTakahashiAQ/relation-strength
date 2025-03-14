import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def create_2d_figure(df_reduced, highlight_indices=None):
    """
    2D散布図を作成してFigureとscatterオブジェクトを返す。
    highlight_indices が指定されていれば該当点を赤枠で強調表示する。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # 点のサイズを小さく (s=10)
    scatter = ax.scatter(
        df_reduced["PC1"],
        df_reduced["PC2"],
        c=df_reduced["Cluster"],
        cmap="tab10",
        s=10,
    )
    for i, txt in enumerate(df_reduced["Data"]):
        ax.annotate(
            txt,
            (df_reduced["PC1"][i], df_reduced["PC2"][i]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
        )
    if highlight_indices:
        # 該当点を赤い輪郭で上書き
        highlight_data = df_reduced.iloc[highlight_indices]
        ax.scatter(
            highlight_data["PC1"],
            highlight_data["PC2"],
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Strength Finder 2D 散布図")
    return fig, scatter


def get_legend_df(df_reduced):
    """
    df_reducedからクラスタごとの色情報を含む凡例用DataFrameを生成する
    """
    unique_clusters = sorted(df_reduced["Cluster"].unique())
    legend_items = []
    for i in unique_clusters:
        rgba = cm.tab10(i)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        legend_items.append({"Cluster": i, "Color": hex_color})
    legend_df = pd.DataFrame(legend_items)
    return legend_df
