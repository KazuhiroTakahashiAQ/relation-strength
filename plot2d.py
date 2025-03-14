import plotly.express as px
import plotly.graph_objects as go


def create_2d_figure(df_reduced, highlight_indices=None):
    """
    Plotlyを用いた2D散布図を作成してFigureオブジェクトを返す。
    - hover_dataで行番号（Data列）を表示
    - highlight_indicesが指定されていれば追加トレースで強調表示する
    - x=0, y=0 のラインを描画して、4象限がわかるようにする
    """
    # 基本の2D散布図（PC1 vs PC2）
    fig = px.scatter(
        df_reduced,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Data"],  # ホバー時に行番号を表示
        title="2D インタラクティブ散布図",
    )
    # マーカーサイズを小さく
    fig.update_traces(marker=dict(size=3))

    # 強調表示用に追加トレースを重ねる
    if highlight_indices:
        highlight_df = df_reduced.iloc[highlight_indices]
        fig.add_trace(
            go.Scatter(
                x=highlight_df["PC1"],
                y=highlight_df["PC2"],
                mode="markers",
                marker=dict(size=8, color="red", symbol="circle-open"),
                name="Selected",  # 凡例上の名前
            )
        )

    # 4象限がわかるように軸の原点ラインを太線で表示
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

    return fig
