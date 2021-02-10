import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def Visualization():

    fig_1 = plt.subplot(121)

    fig_2 = plt.subplot(122)
    pass


def plot_3d(encode, labels):
    fig = plt.figure(num="SDAE")  # 画布
    ax = Axes3D(fig)
    X, Y, Z = encode[:, 0], encode[:, 1], encode[:, 2]
    for x, y, z, s in zip(X, Y, Z, labels):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x=x, y=y, z=z, s=s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()


def plot_2d(predicts, labels):
    fig = plt.figure(num="SDAE")  # 画布
    X, Y = predicts[:, 0], predicts[:, 1]
    C = ['r' if label == 0 else 'b' for label in labels]
    for x, y, c in zip(X, Y, C):
        plt.scatter(x=x, y=y, s=75, c=c, alpha=0.5)
    plt.xlim(X.min(), X.max())
    plt.ylim((Y.min(), Y.max()))


if __name__ == "__main__":
    """Single classifier: validation best epoch = epoch1"""

    # sdae softmax best
    vis = torch.load("../../data/visualization/visualization_sdae-p3-c3-f5.bin")
    predicts, labels = vis["epoch9"]["validation_classifier"], vis["epoch9"]["validation_labels"]

    # single softmax best
    # vis = torch.load("../../data/visualization/visualization_single_softmax.bin")
    # predicts, labels = vis["epoch6"]["validation_classifier"], vis["epoch6"]["validation_labels"]

    # dnn softmax best
    # vis = torch.load("../../data/visualization/visualization_dnn_softmax.bin")
    # predicts, labels = vis["epoch1"]["validation_classifier"], vis["epoch1"]["validation_labels"]

    plot_2d(predicts, labels)
    pass