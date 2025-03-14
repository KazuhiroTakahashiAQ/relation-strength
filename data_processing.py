import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import re

def load_source_data(file_path="data.txt"):
    """
    ファイルからソースデータを読み込み、DataFrameと行リストを返す
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data_str = f.read()
    # 改行で分割し、タブ、スペース、カンマで各項目に分割
    rows = [re.split(r'[\t ,]+', line.strip()) for line in data_str.strip().split("\n")]
    df = pd.DataFrame(rows, columns=["強み1", "強み2", "強み3", "強み4", "強み5"])
    return df, rows


def transform_data(rows):
    """
    読み込んだ行データからワンホットエンコーディングとPCA処理を実施する
    """
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(rows)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(binary_matrix)
    return binary_matrix, reduced


def get_processed_data(file_path="data.txt"):
    """
    ソースデータの読み込みから整形までを一括で行い、元データ、行データ、ワンホット行列、PCA結果を返す
    """
    df, rows = load_source_data(file_path)
    binary_matrix, reduced = transform_data(rows)
    return df, rows, binary_matrix, reduced
