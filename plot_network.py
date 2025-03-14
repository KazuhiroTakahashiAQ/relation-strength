import networkx as nx
import plotly.graph_objects as go


def create_network_graph(rows):
    """
    rows のデータから、各強みの共起ネットワークを構築し、Plotly でインタラクティブなネットワークグラフを作成する。
    """
    G = nx.Graph()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if G.has_edge(row[i], row[j]):
                    G[row[i]][row[j]]["weight"] += 1
                else:
                    G.add_edge(row[i], row[j], weight=1)
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="ネットワークグラフ",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig
