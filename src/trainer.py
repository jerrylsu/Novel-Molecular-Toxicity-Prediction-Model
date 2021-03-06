"""
Triner for Softmax model, DNN model and Capsule Networks Model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import torch
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import numpy as np
from typing import Mapping
sys.path.append(os.path.dirname(os.getcwd()))
from src.utils.metrics import Metrics
from src.featurizers.featurizer import CSFPDataset, get_dataloader
from src.models.softmax_model import SoftmaxModel
from src.models.dnn_model import DNNModel
from src.models.capsule_model import CapsuleModel
from src.utils.utils import custom_collate_fn

PROJECT_DIR = os.path.dirname(os.getcwd())  # get current working directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization')
LOG_DIR = os.path.join(PROJECT_DIR, 'log')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.set_seed(42)
        self.metrics = Metrics()
        self.train_dataset = CSFPDataset(self.args.train_input_file)
        self.validation_dataset = CSFPDataset(self.args.validation_input_file)
        self.train_dataloader, self.validation_dataloader = get_dataloader(train_dataset=self.train_dataset,
                                                                           validation_dataset=self.validation_dataset,
                                                                           collate_fn=custom_collate_fn,
                                                                           batch_size=self.args.batch_size,
                                                                           num_workers=self.args.num_workers,
                                                                           shuffle=True)
        self.train_total, self.validation_total = len(self.train_dataset), len(self.validation_dataset)
        self.train_input_size = next(iter(self.train_dataloader))["input_ids"].shape[1]
        self.validation_input_size = next(iter(self.validation_dataloader))["input_ids"].shape[1]
        if self.args.model_name == "DNN":
            self.classifier_model = DNNModel(input_size=self.train_input_size).to(self.args.device)
        elif self.args.model_name == "Softmax":
            self.classifier_model = SoftmaxModel(input_size=self.train_input_size).to(self.args.device)
        elif self.args.model_name == "Capsule":
            self.classifier_model = CapsuleModel(conv_inputs=1,
                                                 conv_outputs=1,      # 256,
                                                 num_primary_units=8,
                                                 primary_unit_size=8*253,  # fixme get from conv2d  61(128)---253(512)--509(1024)
                                                 num_output_units=2,           # one for each MNIST digit
                                                 output_unit_size=128).to(self.args.device)
        else:
            raise ValueError("Please input the right model type.")
        self.writer = SummaryWriter(self.args.log_path)
        # self.writer.add_graph(model=self.classifier_model,
        #                       input_to_model=next(iter(self.train_dataloader))["input_ids"].to(self.args.device))
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def to_serialization(self, visualization: Mapping):
        if not os.path.exists(self.args.visualization_dir):
            os.mkdir(self.args.visualization_dir)
        torch.save(visualization, os.path.join(self.args.visualization_dir, f"visualization_{self.args.model_name}.pt"))

    def eval(self, epoch):
        self.classifier_model.eval()
        predictions_vis, predictions, labels = [], [], []
        for i, batch in enumerate(tqdm(self.validation_dataloader, desc=f"Eval: ")):
            batch = {key: value.to(self.args.device) for key, value in batch.items()}
            input_ids, label = batch["input_ids"], batch["label"].view(-1)
            with torch.no_grad():
                # Softmax model
                if self.args.model_name == "Capsule":
                    prediction, sdae_encoded = self.classifier_model(input_ids)
                    classifier_model_loss = self.classifier_model.criterion(sdae_encoded, prediction, label)
                    #predict = torch.sqrt((prediction ** 2).sum(dim=2))
                    #classifier_model_loss = self.classifier_model.criterion(predict, label)
                else:
                    prediction = self.classifier_model(input_ids)
                    classifier_model_loss = self.classifier_model.criterion(prediction, label)
                # for tensorboard
                self.writer.add_scalar(tag=f"{self.args.model_name} Model Validation Loss",
                                       scalar_value=classifier_model_loss.item(),
                                       global_step=epoch * len(self.validation_dataloader) + i)
                # for visualization
                predictions_vis.append(prediction.detach())
                labels.append(label)
                # for metric
                if self.args.model_name == "Capsule":
                    prediction = torch.sqrt((prediction ** 2).sum(dim=2)).max(1)[1]
                else:
                    prediction = prediction.data.max(1, keepdim=True)[1]
                predictions.append(prediction)
        predictions_vis = torch.cat(predictions_vis, dim=0).cpu().numpy()
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        validation_recall = f"Recall of validation epoch {epoch}: {round(self.metrics.calculate_recall(labels, predictions), 4)}"
        validation_precision = f"Precision of validation epoch {epoch}: {round(self.metrics.calculate_precision(labels, predictions), 4)}"
        validation_f1 = f"F1 of validation epoch {epoch}: {round(self.metrics.calculate_f1(labels, predictions), 4)}"
        validation_auc = f"Auc of validation epoch {epoch}: {round(self.metrics.calculate_auc(labels, predictions), 4)}"
        validation_accuracy = f"Accuracy of validation epoch {epoch}: {round(self.metrics.calculate_accuracy(labels, predictions), 4)}"
        print(f"Validatopn loss: {round(classifier_model_loss.cpu().item(), 4)}")
        print(validation_accuracy)
        return validation_recall, validation_precision, validation_f1, validation_auc, validation_accuracy, predictions_vis, labels

    def train(self):
        self.set_seed(self.args.seed)
        visualization_data = {}
        for epoch in range(self.args.epochs):
            self.classifier_model.train()
            start_time = time.time()
            predictions_vis, predictions, labels = [], [], []
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}: ")):
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                input_ids, label = batch["input_ids"], batch["label"]
                label = label.view(-1)

                # Classifier model
                if self.args.model_name == "Capsule":
                    prediction, sdae_encoded = self.classifier_model(input_ids)
                    classifier_model_loss = self.classifier_model.criterion(sdae_encoded, prediction, label)
                    #predict = torch.sqrt((prediction ** 2).sum(dim=2))
                    #classifier_model_loss = self.classifier_model.criterion(predict, label)
                else:
                    prediction = self.classifier_model(input_ids)
                    classifier_model_loss = self.classifier_model.criterion(prediction, label)
                self.classifier_model.optimizer.zero_grad()
                classifier_model_loss.backward()
                self.classifier_model.optimizer.step()

                # for tensorboard
                self.writer.add_scalar(tag=f"{self.args.model_name} Model Train Loss",
                                       scalar_value=classifier_model_loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)
                # for visualization
                predictions_vis.append(prediction.detach())
                labels.append(label)
                # for metrics
                if self.args.model_name == "Capsule":
                    prediction = torch.sqrt((prediction ** 2).sum(dim=2)).max(1)[1]
                else:
                    prediction = prediction.data.max(1, keepdim=True)[1]
                predictions.append(prediction)
            predictions_vis = torch.cat(predictions_vis, dim=0).cpu().numpy()
            predictions = torch.cat(predictions, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            train_recall = f"Recall of train epoch {epoch}: {round(self.metrics.calculate_recall(labels, predictions), 4)}"
            train_precision = f"Precision of train epoch {epoch}: {round(self.metrics.calculate_precision(labels, predictions), 4)}"
            train_f1 = f"F1 of train epoch {epoch}: {round(self.metrics.calculate_f1(labels, predictions), 4)}"
            train_auc = f"Auc of train epoch {epoch}: {round(self.metrics.calculate_auc(labels, predictions), 4)}"
            train_accuracy = f"Accuracy of train epoch {epoch}: {round(self.metrics.calculate_accuracy(labels, predictions), 4)}"
            validation_recall, validation_precision, validation_f1, validation_auc, validation_accuracy, validation_predictions_vis, validation_labels = self.eval(epoch=epoch)
            print(f"Train loss: {round(classifier_model_loss.cpu().item(), 4)}")
            print(train_accuracy)
            visualization_data[f"epoch{epoch}"] = {"train_classifier": predictions_vis,
                                                   "train_labels": labels,
                                                   "train_recall": train_recall,
                                                   "train_precision": train_precision,
                                                   "train_f1": train_f1,
                                                   "train_auc": train_auc,
                                                   "train_accuracy": train_accuracy,
                                                   "validation_classifier": validation_predictions_vis,
                                                   "validation_labels": validation_labels,
                                                   "validation_recall": validation_recall,
                                                   "validation_precision": validation_precision,
                                                   "validation_f1": validation_f1,
                                                   "validation_auc": validation_auc,
                                                   "validation_accuracy": validation_accuracy}

            total_time = time.time() - start_time
        # save for visualization
        self.to_serialization(visualization_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_input_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'dataset/train_file.pt'),
                        help="Path of the train dataset.")
    parser.add_argument("--train_label_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'LabelTrain.csv'),
                        help="Path of the train label file.")
    parser.add_argument("--validation_input_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'dataset/validate_file.pt'),
                        help="Path of the validation dataset.")
    parser.add_argument("--validation_label_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'LabelValidation.csv'),
                        help="Path of the validation label file.")
    parser.add_argument("--visualization_dir",
                        type=str,
                        default=VISUALIZATION_DIR,
                        help="Output for visualization.")
    parser.add_argument("--model_name",
                        type=str,
                        default="Capsule",  # or Softmax / DNN / Capsule
                        help="Model name.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--log_path", type=str, default=LOG_DIR, help="Path of the log.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--classifier_lr", type=float, default=0.01, help="Learning rate of the Classifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of subprocesses for data loading.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="The steps of warm up.")
    args = parser.parse_args()
    trainer = Trainer(args=args)
    trainer.train()
    pass
