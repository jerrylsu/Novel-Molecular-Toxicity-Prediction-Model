import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(num="SDAE")
ax = Axes3D(fig)


def plot_3d(encode, labels):
    X, Y, Z = encode.data[:, 0].numpy(), encode.data[:, 1].numpy(), encode.data[:, 2].numpy()
    labels = labels.numpy()
    for x, y, z, s in zip(X, Y, Z, labels):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x=x, y=y, z=z, s=s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()


if __name__ == "__main__":
    fig = plt.figure(num='SDAE')
    ax = Axes3D(fig)
    pass