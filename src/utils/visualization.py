import torch
from itertools import cycle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import roc_curve, auc


def plot_roc(predicts, labels, name: str):
    """Compute and plot ROC & AUC.
    predicts: [batch_size, n_classes]
    labels: [batch_size,]
    name: string
    """
    # Compute ROC curve and ROC area for each class
    labels = torch.nn.functional.one_hot(torch.from_numpy(labels), num_classes=2).numpy()
    n_classes = predicts.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predicts[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predicts.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'micro-average ROC curve (area = {round(roc_auc["micro"], 2)})')
    colors = cycle(['aqua', 'darkorange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=2, label=f'ROC curve of class {i} (area = {round(roc_auc[i], 2)})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    plt.show()


def Visualization():

    fig_1 = plt.subplot(121)

    fig_2 = plt.subplot(122)
    pass


def plot_3d(encode, labels, name: str):
    fig = plt.figure(num=name)  # 画布
    ax = Axes3D(fig)
    X, Y, Z = encode[:, 0], encode[:, 1], encode[:, 2]
    for x, y, z, s in zip(X, Y, Z, labels):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x=x, y=y, z=z, s=s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()


def plot_2d(predicts, labels, name: str):
    fig = plt.figure(num=name)
    X, Y = predicts[:, 0], predicts[:, 1]
    C = ['r' if label == 0 else 'b' for label in labels]
    for x, y, c in zip(X, Y, C):
        plt.scatter(x=x, y=y, s=75, c=c, alpha=0.5)
    plt.xlim(X.min(), X.max())
    plt.ylim((Y.min(), Y.max()))


if __name__ == "__main__":
    """Classifier Model: Softmax, DNN, SDAE"""
    # softmax model
    softmax_vis = torch.load("../../data/visualization/visualization_Softmax.pt")
    softmax_predicts, softmax_labels = softmax_vis["epoch4"]["validation_classifier"], softmax_vis["epoch4"]["validation_labels"]

    # dnn model
    dnn_vis = torch.load("../../data/visualization/visualization_DNN.pt")
    dnn_predicts, dnn_labels = dnn_vis["epoch1"]["validation_classifier"], dnn_vis["epoch1"]["validation_labels"]

    # sdae model
    sdae_vis = torch.load("../../data/visualization/visualization_SDAE-p3-c3-f5_ext.pt")
    sdae_predicts, sdae_labels = sdae_vis["epoch1"]["validation_classifier"], sdae_vis["epoch1"]["validation_labels"]

    # capsule model
    capsule_vis = torch.load("../../data/visualization/visualization_Capsule_ext.pt")
    capsule_predicts, capsule_labels = capsule_vis["epoch2"]["validation_classifier"], capsule_vis["epoch2"]["validation_labels"]
    capsule_predicts = capsule_predicts.sum(axis=-1)

    plot_roc(softmax_predicts, softmax_labels, "Softmax Model")
    plot_roc(dnn_predicts, dnn_labels, "SSAE+Softmax Model")
    plot_roc(sdae_predicts, sdae_labels, "SSAE+Capsule Model")

    plot_2d(capsule_predicts, capsule_labels)
    plot_2d(sdae_predicts, sdae_labels)
    plot_2d(softmax_predicts, softmax_labels)
    plot_2d(dnn_predicts, dnn_labels)
    pass