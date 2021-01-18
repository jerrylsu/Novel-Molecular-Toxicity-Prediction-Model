import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import torch
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from torch.autograd import Variable
sys.path.append(os.path.dirname(os.getcwd()))
from src.featurizers.featurizer import CSFPDataset, get_dataloader
from src.models.model import StackedAutoEncoderModel, ClassifierLayer

PROJECT_PATH = os.path.dirname(os.getcwd())  # get current working directory
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
LOG_PATH = os.path.join(PROJECT_PATH, 'log')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_dataset = CSFPDataset(self.args.input_file, self.args.label_file)
        self.train_dataloader, self.test_dataloader = get_dataloader(train_dataset=self.train_dataset,
                                                                     batch_size=self.args.batch_size)
        self.sae_model = StackedAutoEncoderModel().to(self.args.device)
        self.classifier = ClassifierLayer().to(self.args.device)
        self.writer = SummaryWriter(self.args.log_path)
        # self.writer.add_graph(model=self.sae_model, input_to_model=next(iter(self.train_dataloader))["input_ids"])
        pass

    def train(self):
        for epoch in range(self.args.epochs):
            #if epoch % 10 == 0:
                # Test the quality of our features with a randomly initialzed linear classifier.
                #self.classifier = ClassifierLayer().to(self.args.device)
            self.sae_model.train()
            total_time = time.time()
            correct = 0
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch}: ")):
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                input_ids, label = batch["input_ids"], batch["label"]
                output = self.sae_model(input_ids)
                output_ids, sae_loss = output[0].detach(), output[1]
                output_ids = output_ids.view(output_ids.size(0), -1)
                label = label.view(-1)
                prediction = self.classifier(output_ids)
                classifier_loss = self.classifier.criterion(prediction, label)

                self.classifier.optimizer.zero_grad()
                classifier_loss.backward()
                self.classifier.optimizer.step()
                self.writer.add_scalar(tag="Classifier Train Loss",
                                       scalar_value=classifier_loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)
                self.writer.add_scalar(tag="SAE Train Loss",
                                       scalar_value=sae_loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)

                pred = torch.argmax(prediction.data, dim=-1)
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()

            total_time = time.time() - total_time

        features, labels = [], []
        self.sae_model.eval()
        for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Eval: ")):
            batch = {key: value.to(self.args.device) for key, value in batch.items()}
            input_ids, label = batch["input_ids"], batch["label"]
            with torch.no_grad():
                output = self.sae_model(input_ids)
                output_ids, sae_loss = output[0].detach(), output[1]
                output_ids = output_ids.view(output_ids.size(0), -1)
                label = label.view(-1)
                features.append(output_ids)
                labels.append(label)
                features = torch.stack(features)
        # self.writer.add_embedding(mat=features, metadata=labels, label_img=input_ids)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default=os.path.join(DATA_PATH, 'train.csv'),
                        help="Path of the dataset.")
    parser.add_argument("--label_file",
                        type=str,
                        default=os.path.join(DATA_PATH, 'Jiang1823Train.csv'),
                        help="Path of the label file.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--log_path", type=str, default=LOG_PATH, help="Path of the log.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--classifier_lr", type=float, default=0.001, help="Learning rate of the Classifier.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses for data loading.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="The steps of warm up.")
    args = parser.parse_args()
    trainer = Trainer(args=args)
    trainer.train()
    pass
