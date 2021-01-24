import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src.train import VISUALIZATION_DIR


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
    visualization_data = torch.load("/Users/jerry/PycharmProjects/SDAE/data/visualization/visualization_exp1.bin")
    sdae, classifer, labels = visualization_data["epoch33"]["sdae"], visualization_data["epoch33"]["classifier"], visualization_data["epoch33"]["labels"]
    #plot_3d(sdae, labels)
    plot_2d(classifer, labels)
    pass