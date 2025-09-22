import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification


# from umap import UMAP

def generate_unique_colormap(labels):
    unique_labels = sorted(np.unique(labels))
    num_labels = len(unique_labels)

    # 生成HSV颜色参数
    hues = np.linspace(0, 1, num_labels, endpoint=False) * 2
    hues = hues - np.floor(hues)
    saturations = np.asarray(list(range(num_labels)), dtype=int) % 4 * 0.1 + 0.4  # 固定高饱和度
    values = np.asarray(list(range(num_labels)), dtype=int) % 5 * 0.1 + 0.4  # 固定高亮度

    # 转换为RGB颜色
    hsv_colors = np.column_stack([hues, saturations, values])
    rgb_colors = hsv_to_rgb(hsv_colors)

    # 创建颜色字典
    color_dict = {label: rgb_colors[i] for i, label in enumerate(unique_labels)}

    return ListedColormap(rgb_colors), color_dict


# 数据降维
def reduce_dimension(features, method='tsne', pca_components=0.95):
    """
    带自动PCA降噪的维度缩减流程
    Args:
        features (np.ndarray): 输入特征矩阵，形状为 (样本数, 特征维度)
        method (str): 降维方法，支持 'tsne' 或 'umap'
        pca_components: PCA保留的方差比例(0-1)或组件数，默认保留95%方差
    Returns:
        np.ndarray: 降维后的二维坐标
    """
    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 自动PCA降维
    pca = PCA(n_components=pca_components)
    pca_features = pca.fit_transform(features_scaled)
    print(f"PCA保留维度: {pca.n_components_} (解释方差: {pca.explained_variance_ratio_.sum():.2%})")

    # 选择降维方法
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=30,  # 根据数据量调整(5-50)
            random_state=11
        )
    # elif method == 'umap':
    #     reducer = UMAP(
    #         n_components=2,
    #         n_neighbors=15,  # 小数据集建议减小该值
    #         min_dist=0.1,
    #         random_state=42
    #     )
    else:
        raise ValueError("支持的降维方法: 'tsne' 或 'umap'")

    return reducer.fit_transform(pca_features)


def plot_embeddings(embeddings, labels, length, mode ,title, save_path):
    # 数据校验
    assert len(embeddings) == len(labels), "特征与标签数量不一致！"
    labels = np.asarray(labels)

    # 获取唯一标签并排序
    unique_labels = sorted(np.unique(labels))
    num_classes = len(unique_labels)
    print(f'num_class: {num_classes}')

    # # 创建固定颜色映射
    # base_colors = plt.cm.tab20.colors  # 支持最多20种固定颜色
    # repeat_times = (num_classes // 20) + 1
    # color_list = (base_colors * repeat_times)[:num_classes]
    # cmap = ListedColormap(color_list)
    cmap, color_dict = generate_unique_colormap(labels)

    # 标签编码(确保颜色与标签的固定对应关系)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # 创建画布
    plt.figure(figsize=(12, 8), dpi=100)

    # 绘制散点图
    scatter = plt.scatter(
        embeddings[:length, 0],
        embeddings[:length, 1],
        c=encoded_labels[:length],
        cmap=cmap,
        alpha=0.8,
        s=10,
        # edgecolor='w',
        linewidth=0.5
    )

    # center = plt.scatter(
    #     embeddings[length:length + num_classes, 0],
    #     embeddings[length:length + num_classes, 1],
    #     c=encoded_labels[length:length + num_classes],
    #     cmap=cmap,
    #     alpha=0.8,
    #     s=50,
    #     edgecolor='gold',
    #     linewidth=2,
    #     marker='^'
    # )

    for i, (x, y) in enumerate(zip(embeddings[length:length + num_classes], labels[length:length + num_classes])):
        plt.text(
            x[0], x[1],
            str(y),  # 直接显示原始标签
            color='black',  # 使用对应类别的颜色
            fontsize=8,  # 可调节字体大小
            ha='center',  # 水平居中
            va='center',  # 垂直居中
            alpha=0.9
        )

    # center_box = plt.scatter(
    #     embeddings[length + num_classes:, 0],
    #     embeddings[length + num_classes:, 1],
    #     c=encoded_labels[length + num_classes:],
    #     cmap=cmap,
    #     alpha=0.8,
    #     s=50,
    #     edgecolor='black',
    #     linewidth=2,
    #     marker='s'
    # )

    if mode == 'box':
        for i, (x, y) in enumerate(zip(embeddings[length + num_classes:], labels[length + num_classes:])):
            plt.text(
                x[0], x[1],
                str(y),  # 直接显示原始标签
                color='red',  # 使用对应类别的颜色
                fontsize=8,  # 可调节字体大小
                ha='center',  # 水平居中
                va='center',  # 垂直居中
                alpha=0.9
            )

    # # 创建图例项(显示所有类别，即使某些类别没有数据点)
    # legend_items = []
    # for idx, label in enumerate(unique_labels):
    #     count = np.sum(labels == label)
    #     legend_items.append(
    #         mpatches.Patch(
    #             color=color_list[idx],
    #             label=f"{label} (n={count})"  # 显示类别名称和样本数量
    #         )
    #     )
    #
    # # 添加图例
    # plt.legend(
    #     handles=legend_items,
    #     title="Categories",
    #     bbox_to_anchor=(1.05, 1),
    #     loc='upper left',
    #     borderaxespad=0.
    # )

    # 图表装饰
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
