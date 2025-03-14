import plotly.express as px


def create_3d_figure(df_reduced, highlight_indices=None):
    """
    Plotly Express を用いてインタラクティブな 3D 散布図を作成して Figure を返す。
    highlight_indices が指定されていれば、該当点を追加トレースで強調表示する。
    """
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
    # マーカーサイズをさらに小さく (size=3)
    fig.update_traces(marker=dict(size=3))
    if highlight_indices:
        highlight_df = df_reduced.iloc[highlight_indices]
        fig.add_scatter3d(
            x=highlight_df["PC1"],
            y=highlight_df["PC2"],
            z=highlight_df["PC3"],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle-open"),
            name="Selected",
        )
    return fig
