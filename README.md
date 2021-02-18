## Novel Molecular Toxicity Prediction Model

Novel molecular toxicity prediction model based on Softmax / Deep Neural Network / Stacked Autoencoder / Stacked Capsule Model.

| Model Name | Recall | Precision  | F1 | AUC | Accuracy |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Softmax Model | 0.7235 | 0.8601 | 0.7859 | 0.8100 | 0.8154 |
| Deep Neural Network Model | 0.7765 | 0.8354 | 0.8049 | 0.8209| 0.8237 |
| Stacked Autoencoder Model | 0.7706 | 0.8562 | 0.8111 | 0.8283 | 0.8320 |
| Stacked Capsule Network Model | | | | | |

### 1. Softmax Model

- loss

![loss](./data/results/softmax/loss.png)

- validationset classification

![validationset](./data/results/softmax/validation_best.png)

### 2. Deep Neural Network Model

- loss

![loss](./data/results/deep_neural_network/loss.jpeg)

- trainset classification

![tainset](./data/results/deep_neural_network/train_epoch4.png)

- validationset classification

![validationset](./data/results/deep_neural_network/validation_best.png)

### 3. Stacked Autoencoder

- loss

![loss](./data/results/stacked_autoencoder/loss.png)

- validationset classification

![validationset](./data/results/stacked_autoencoder/validation_best.png)

### Reference

- Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. Dynamic routing between capsules. In NIPS, pages 3859-3869, 2017.

- Wang, Y.-W., Huang, L., Jiang, S.-W., Li, K., Zou, J., & Yang, S.-Y. (2020). CapsCarcino: A novel sparse data deep learning tool for predicting carcinogens. Food and Chemical Toxicology, 135, 110921.

