import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import torch
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from torch.autograd import Variable
import numpy as np
from typing import Mapping
sys.path.append(os.path.dirname(os.getcwd()))
from src.featurizers.featurizer import CSFPDataset, get_dataloader
from src.models.model import StackedAutoEncoderModel, ClassifierLayer
from src.utils.utils import custom_collate_fn

PROJECT_DIR = os.path.dirname(os.getcwd())  # get current working directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization')
LOG_DIR = os.path.join(PROJECT_DIR, 'log')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_dataset = CSFPDataset(self.args.train_input_file, self.args.train_label_file)
        # self.validation_dataset = CSFPDataset(self.args.validation_input_file, self.args.validation_label_file)
        self.train_dataloader, self.validation_dataloader = get_dataloader(train_dataset=self.train_dataset,
                                                                           # validation_dataset=self.validation_dataset,
                                                                           collate_fn=custom_collate_fn,
                                                                           batch_size=self.args.batch_size,
                                                                           num_workers=self.args.num_workers,
                                                                           shuffle=True)
        self.train_total = len(self.train_dataset)
        self.train_input_size = next(iter(self.train_dataloader))["input_ids"].shape[1]
        self.sdae_model = StackedAutoEncoderModel(input_size=self.train_input_size, output_size=3).to(self.args.device)
        self.classifier = ClassifierLayer().to(self.args.device)
        self.writer = SummaryWriter(self.args.log_path)
        self.writer.add_graph(model=self.sdae_model,
                              input_to_model=next(iter(self.train_dataloader))["input_ids"].to(self.args.device))
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
        torch.save(visualization, os.path.join(self.args.visualization_dir, "visualization.bin"))

    def eval(self):
        features, labels = [], []
        self.sdae_model.eval()
        for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Eval: ")):
            batch = {key: value.to(self.args.device) for key, value in batch.items()}
            input_ids, label = batch["input_ids"], batch["label"]
            with torch.no_grad():
                sdae_output = self.sdae_model(input_ids)
                encoded, decoded = sdae_output[0].detach(), sdae_output[1]
                encoded = encoded.view(encoded.size(0), -1)
        # self.writer.add_embedding(mat=features, metadata=labels, label_img=input_ids)

    def train(self):
        self.set_seed(self.args.seed)
        visualization_data = {}
        for epoch in range(self.args.epochs):
            #if epoch % 10 == 0:
                # Test the quality of our features with a randomly initialzed linear classifier.
                #self.classifier = ClassifierLayer().to(self.args.device)
            self.sdae_model.train()
            start_time = time.time()
            correct = 0
            sdae_epoch_data, classifier_epoch_data, labels = [], [], []
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}: ")):
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                input_ids, label = batch["input_ids"], batch["label"]
                sdae_outputs = self.sdae_model(input_ids)
                sdae_encoded, sae_loss = sdae_outputs[0], sdae_outputs[1]
                sdae_encoded = sdae_encoded.view(sdae_encoded.size(0), -1)
                label = label.view(-1)
                prediction = self.classifier(sdae_encoded)
                classifier_loss = self.classifier.criterion(prediction, label)
                self.classifier.optimizer.zero_grad()
                classifier_loss.backward()
                self.classifier.optimizer.step()
                # for tensorboard
                self.writer.add_scalar(tag="Classifier Train Loss",
                                       scalar_value=classifier_loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar(tag="SAE Train Loss",
                                       scalar_value=sae_loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)
                # for visualization
                sdae_epoch_data.append(sdae_encoded.detach())
                classifier_epoch_data.append(prediction.detach())
                labels.append(label)
                # for metrics
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).cpu().numpy().sum()
            accuracy = f"Accuracy of epoch {epoch}: {round(correct / self.train_total, 4)}"
            visualization_data[f"epoch{epoch}"] = {"sdae": torch.cat(sdae_epoch_data, dim=0).cpu().numpy(),
                                                   "classifier": torch.cat(classifier_epoch_data, dim=0).cpu().numpy(),
                                                   "labels": torch.cat(labels, dim=0).cpu().numpy(),
                                                   "accuracy": accuracy}
            total_time = time.time() - start_time
        # save for visualization
        self.to_serialization(visualization_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_input_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'train_jerry.binary.csv'),
                        help="Path of the train dataset.")
    parser.add_argument("--train_label_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'LabelTrain.csv'),
                        help="Path of the train label file.")
    parser.add_argument("--validation_input_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'validation_jerry.binary.csv'),
                        help="Path of the validation dataset.")
    parser.add_argument("--validation_label_file",
                        type=str,
                        default=os.path.join(DATA_DIR, 'ValidationTrain.csv'),
                        help="Path of the validation label file.")
    parser.add_argument("--visualization_dir",
                        type=str,
                        default=VISUALIZATION_DIR,
                        help="Output for visualization.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--log_path", type=str, default=LOG_DIR, help="Path of the log.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--classifier_lr", type=float, default=0.001, help="Learning rate of the Classifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses for data loading.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="The steps of warm up.")
    args = parser.parse_args()
    trainer = Trainer(args=args)
    trainer.train()
    pass
