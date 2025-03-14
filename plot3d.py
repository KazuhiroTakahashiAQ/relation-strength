import plotly.express as px
import plotly.graph_objects as go


def create_3d_figure(df_reduced, highlight_indices=None):
    """
    Plotly Express を用いてインタラクティブな 3D 散布図を作成して Figure を返す。
    - hover_data で行番号（Data列）を表示
    - highlight_indices が指定されていれば、該当点を追加トレースで強調表示する。
    - x=0, y=0, z=0 の原点ラインを可視化 (zeroline=True) して象限がわかるようにする
      (3Dの場合は厳密には8象限になる)
    """
    fig = px.scatter_3d(
        df_reduced,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        hover_data=["Data"],
        title="3D インタラクティブ散布図",
        width=800,
        height=600,
    )
    # マーカーサイズを小さく (size=3)
    fig.update_traces(marker=dict(size=2))

    # 強調表示用に追加トレース
    if highlight_indices:
        highlight_df = df_reduced.iloc[highlight_indices]
        fig.add_trace(
            go.Scatter3d(
                x=highlight_df["PC1"],
                y=highlight_df["PC2"],
                z=highlight_df["PC3"],
                mode="markers",
                marker=dict(size=8, color="red", symbol="circle-open"),
                name="Selected",
            )
        )

    # 3D軸で x=0, y=0, z=0 のラインを引く代わりに
    # Plotly の zeroline 設定で原点ラインを表示
    fig.update_layout(
        scene=dict(
            xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=2),
            yaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=2),
            zaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=2),
        )
    )

    return fig
