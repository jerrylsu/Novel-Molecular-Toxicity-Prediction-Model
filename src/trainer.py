import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from typing import Any, Callable, Optional, Mapping
import torch
import torch.nn.functional as F
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import SGD
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.utils.utils import custom_collate_fn
from torch.utils.tensorboard import SummaryWriter
from src.models.sdae_model import AutoencoderLayer, StackedAutoEncoderModel
from src.models.classifier_model import ClassifierModel
from src.featurizers.featurizer import CSFPDataset, get_dataloader

PROJECT_DIR = os.path.dirname(os.getcwd())  # get current working directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization')
LOG_DIR = os.path.join(PROJECT_DIR, 'log')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        #self.train_dataset = CSFPDataset(self.args.train_input_file)
        #self.validation_dataset = CSFPDataset(self.args.validation_input_file)
        #self.train_dataloader, self.validation_dataloader = get_dataloader(train_dataset=self.train_dataset,
        #                                                                   validation_dataset=self.validation_dataset,
        #                                                                   collate_fn=custom_collate_fn,
        #                                                                   batch_size=self.args.batch_size,
        #                                                                   num_workers=self.args.num_workers,
        #                                                                   shuffle=True)
        #self.train_total, self.validation_total = len(self.train_dataset), len(self.validation_dataset)
        #self.train_input_size = next(iter(self.train_dataloader))["input_ids"].shape[1]
        #self.validation_input_size = next(iter(self.validation_dataloader))["input_ids"].shape[1]
        #self.sdae_model = StackedAutoEncoderModel(dimensions=[self.train_input_size, 2048, 1024, 512, 256, 128], final_activation=None).to(self.args.device)
        self.classifier_model = ClassifierModel(input_size=128).to(self.args.device)
        self.writer = SummaryWriter(self.args.log_path)
        # self.writer.add_graph(model=self.sdae_model,
        #                      input_to_model=next(iter(self.train_dataloader))["input_ids"].to(self.args.device))
        pass

    def training_callback(self, epoch, lr, loss, validation_loss):
        self.writer.add_scalars("data/autoencoder",
                                {"lr": lr, "loss": loss, "validation_loss": validation_loss, },
                                epoch,)

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

    def train_sdae_model(
            self,
            dataset: torch.utils.data.Dataset,
            autoencoder: torch.nn.Module,
            epochs: int,
            batch_size: int,
            optimizer: torch.optim.Optimizer,
            scheduler: Any = None,
            validation: Optional[torch.utils.data.Dataset] = None,
            corruption: Optional[float] = None,
            sampler: Optional[torch.utils.data.sampler.Sampler] = None,
            silent: bool = False,
            update_freq: Optional[int] = 1,
            update_callback: Optional[Callable[[float, float], None]] = None,
            num_workers: Optional[int] = None,
            epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    ) -> None:
        """
        Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
        (feature, prediction), then the prediction is stripped away.
        :param dataset: training Dataset, consisting of tensors shape [batch_size, features]
        :param autoencoder: autoencoder to train
        :param epochs: number of training epochs
        :param batch_size: batch size for training
        :param optimizer: optimizer to use
        :param scheduler: scheduler to use, or None to disable, defaults to None
        :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
        :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
        :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
        :param silent: set to True to prevent printing out summary statistics, defaults to False
        :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
        :param update_callback: optional function of loss and validation loss to update
        :param num_workers: optional number of workers to use for data loading
        :param epoch_callback: optional function of epoch and model
        :return: None
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=True if sampler is None else False,
            num_workers=num_workers if num_workers is not None else 0,
        )
        if validation is not None:
            validation_loader = DataLoader(
                validation,
                batch_size=batch_size,
                pin_memory=False,
                sampler=None,
                shuffle=False,
            )
        else:
            validation_loader = None
        loss_function = nn.MSELoss()
        autoencoder.train()
        validation_loss_value = -1
        loss_value = 0
        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()
            data_iterator = tqdm(
                dataloader,
                leave=True,
                unit="batch",
                postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1, },
                disable=silent,
            )
            for index, batch in enumerate(data_iterator):
                if isinstance(batch, dict):
                    batch = {key: value.to(self.args.device) for key, value in batch.items()}
                    input_ids, label = batch["input_ids"], batch["label"]
                elif isinstance(batch, list):
                    input_ids = batch[0].to(self.args.device)
                # run the batch through the autoencoder and obtain the output
                if corruption is not None:
                    output = autoencoder(F.dropout(input_ids, corruption))
                else:
                    output = autoencoder(input_ids)
                loss = loss_function(output, input_ids)
                # accuracy = pretrain_accuracy(output, batch)
                loss_value = float(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                data_iterator.set_postfix(
                    epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
                )
                # for tensorboard
                self.writer.add_scalar(tag="SDAE Train Loss",
                                       scalar_value=loss_value,
                                       global_step=epoch * len(dataloader) + index)
                self.writer.add_scalar(tag="SDAE Val Loss",
                                       scalar_value=validation_loss_value,
                                       global_step=epoch * len(validation_dataset) + index)
            if update_freq is not None and epoch % update_freq == 0:
                if validation_loader is not None:
                    validation_output = self.predict(
                        validation,
                        autoencoder,
                        batch_size,
                        silent=True,
                        encode=False,
                    )
                    validation_inputs = []
                    for val_batch in validation_loader:
                        if isinstance(val_batch, dict):
                            val_batch = {key: value for key, value in val_batch.items()}
                            input_ids, label = val_batch["input_ids"], val_batch["label"]
                        elif isinstance(val_batch, list):
                            input_ids = val_batch[0].to(self.args.device)
                        validation_inputs.append(input_ids)
                    validation_actual = torch.cat(validation_inputs)
                    validation_actual = validation_actual.to(self.args.device)
                    validation_output = validation_output.to(self.args.device)
                    validation_loss = loss_function(validation_output, validation_actual)
                    # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                    validation_loss_value = float(validation_loss.item())
                    data_iterator.set_postfix(
                        epo=epoch,
                        lss="%.6f" % loss_value,
                        vls="%.6f" % validation_loss_value,
                    )
                    autoencoder.train()
                else:
                    validation_loss_value = -1
                    # validation_accuracy = -1
                    data_iterator.set_postfix(
                        epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                    )
                if update_callback is not None:
                    update_callback(
                        epoch,
                        optimizer.param_groups[0]["lr"],
                        loss_value,
                        validation_loss_value,
                    )
            if epoch_callback is not None:
                autoencoder.eval()
                epoch_callback(epoch, autoencoder)
                autoencoder.train()

    def pretrain_ae_model(
            self,
            dataset,
            autoencoder: StackedAutoEncoderModel,
            epochs: int,
            batch_size: int,
            optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
            scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
            validation: Optional[torch.utils.data.Dataset] = None,
            corruption: Optional[float] = None,
            sampler: Optional[torch.utils.data.sampler.Sampler] = None,
            silent: bool = False,
            update_freq: Optional[int] = 1,
            update_callback: Optional[Callable[[float, float], None]] = None,
            num_workers: Optional[int] = None,
            epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    ) -> None:
        """
        Given an autoencoder, train it using the data provided in the dataset; for simplicity the accuracy is reported only
        on the training dataset. If the training dataset is a 2-tuple or list of (feature, prediction), then the prediction
        is stripped away.
        :param dataset: instance of Dataset to use for training
        :param autoencoder: instance of an autoencoder to train
        :param epochs: number of training epochs
        :param batch_size: batch size for training
        :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
        :param optimizer: function taking model and returning optimizer
        :param scheduler: function taking optimizer and returning scheduler, or None to disable
        :param validation: instance of Dataset to use for validation
        :param sampler: sampler to use in the DataLoader, defaults to None
        :param silent: set to True to prevent printing out summary statistics, defaults to False
        :param update_freq: frequency of batches with which to update counter, None disables, default 1
        :param update_callback: function of loss and validation loss to update
        :param num_workers: optional number of workers to use for data loading
        :param epoch_callback: function of epoch and model
        :return: None
        """
        current_dataset = dataset
        current_validation = validation
        number_of_subautoencoders = len(autoencoder.dimensions) - 1
        for index in range(number_of_subautoencoders):
            encoder, decoder = autoencoder.get_stack(index)
            embedding_dimension = autoencoder.dimensions[index]
            hidden_dimension = autoencoder.dimensions[index + 1]
            # manual override to prevent corruption for the last subautoencoder
            if index == (number_of_subautoencoders - 1):
                corruption = None
            # initialise the subautoencoder
            sub_autoencoder = AutoencoderLayer(
                embedding_dimension=embedding_dimension,
                hidden_dimension=hidden_dimension,
                activation=torch.nn.ReLU()
                if index != (number_of_subautoencoders - 1)
                else None,
                corruption=nn.Dropout(corruption) if corruption is not None else None,
            ).to(self.args.device)
            ae_optimizer = optimizer(sub_autoencoder)
            ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler
            self.train_sdae_model(
                current_dataset,
                sub_autoencoder,
                epochs,
                batch_size,
                ae_optimizer,
                validation=current_validation,
                corruption=None,  # already have dropout in the DAE
                scheduler=ae_scheduler,
                sampler=sampler,
                silent=silent,
                update_freq=update_freq,
                update_callback=update_callback,
                num_workers=num_workers,
                epoch_callback=epoch_callback,
            )
            # copy the weights
            sub_autoencoder.copy_weights(encoder, decoder)
            # pass the dataset through the encoder part of the subautoencoder
            if index != (number_of_subautoencoders - 1):
                current_dataset = TensorDataset(
                    self.predict(
                        current_dataset,
                        sub_autoencoder,
                        batch_size,
                        silent=silent,
                    )
                )
                if current_validation is not None:
                    current_validation = TensorDataset(
                        self.predict(
                            current_validation,
                            sub_autoencoder,
                            batch_size,
                            silent=silent,
                        )
                    )
            else:
                current_dataset = None  # minor optimisation on the last subautoencoder
                current_validation = None

    def predict(
            self,
            dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            batch_size: int,
            silent: bool = False,
            encode: bool = True,
    ) -> torch.Tensor:
        """
        Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
        output.
        :param dataset: evaluation Dataset
        :param model: autoencoder for prediction
        :param batch_size: batch size
        :param cuda: whether CUDA is used, defaults to True
        :param silent: set to True to prevent printing out summary statistics, defaults to False
        :param encode: whether to encode or use the full autoencoder
        :return: predicted features from the Dataset
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent, )
        features = []
        if isinstance(model, torch.nn.Module):
            model.eval()
        for batch in data_iterator:
            if isinstance(batch, dict):
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                input_ids, label = batch["input_ids"], batch["label"]
            elif isinstance(batch, list):
                input_ids = batch[0].to(self.args.device)
            if encode:
                output = model.encode(input_ids)
            else:
                output = model(input_ids)
            features.append(output.detach().cpu())  # move to the CPU to prevent out of memory on the GPU
        return torch.cat(features)

    def train_classifier(self,
                         dataset,
                         autoencoder: torch.nn.Module,
                         batch_size,
                         validation: Optional[torch.utils.data.Dataset] = None,
                         sampler: Optional[torch.utils.data.sampler.Sampler] = None,
                         num_workers: Optional[int] = None,
                         ):
        visualization_data = {}
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=True if sampler is None else False,
            num_workers=num_workers if num_workers is not None else 0,
        )
        autoencoder.eval()
        for epoch in range(self.args.classifier_epochs):
            self.classifier_model.train()
            correct = 0
            sdae_epoch_data, train_classifier_epoch_data, train_labels = [], [], []
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}: ")):
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                input_ids, label = batch["input_ids"], batch["label"].view(-1)
                with torch.no_grad():
                    input_ids = autoencoder.encoder(input_ids)

                # Single classifier model
                prediction = self.classifier_model(input_ids)
                classifier_model_loss = self.classifier_model.criterion(prediction, label)
                self.classifier_model.optimizer.zero_grad()
                classifier_model_loss.backward()
                self.classifier_model.optimizer.step()

                # for tensorboard
                self.writer.add_scalar(tag="ClassifierModelFromSDAE Train Loss",
                                       scalar_value=classifier_model_loss.item(),
                                       global_step=epoch * len(dataloader) + i)
                # for visualization
                train_classifier_epoch_data.append(prediction.detach())
                train_labels.append(label)
                # for metrics
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).cpu().numpy().sum()
            train_accuracy = f"Accuracy of train epoch {epoch}: {round(correct / len(dataloader), 4)}"
            validation_accuracy, validation_classifier_epoch_data, validation_labels = self.eval_classifier(epoch=epoch,
                                                                                                            autoencoder=autoencoder,
                                                                                                            batch_size=batch_size,
                                                                                                            validation=validation)
            visualization_data[f"epoch{epoch}"] = {"train_classifier": torch.cat(train_classifier_epoch_data, dim=0).cpu().numpy(),
                                                   "train_labels": torch.cat(train_labels, dim=0).cpu().numpy(),
                                                   "train_accuracy": train_accuracy,
                                                   "validation_classifier": torch.cat(validation_classifier_epoch_data, dim=0).cpu().numpy(),
                                                   "validation_labels": torch.cat(validation_labels, dim=0).cpu().numpy(),
                                                   "validation_accuracy": validation_accuracy}
        # save for visualization
        self.to_serialization(visualization_data)

    def eval_classifier(self,
                        epoch,
                        autoencoder: torch.nn.Module,
                        batch_size,
                        validation: Optional[torch.utils.data.Dataset] = None,):
        self.classifier_model.eval()
        dataloader = DataLoader(
            validation, batch_size=batch_size, pin_memory=False, shuffle=False
        )
        correct = 0
        validation_classifier_epoch_data, validation_labels = [], []
        for i, batch in enumerate(tqdm(dataloader, desc=f"Eval: ")):
            batch = {key: value.to(self.args.device) for key, value in batch.items()}
            input_ids, label = batch["input_ids"], batch["label"]
            with torch.no_grad():
                # Single classifier model
                input_ids = autoencoder.encoder(input_ids)
                prediction = self.classifier_model(input_ids)
                classifier_model_loss = self.classifier_model.criterion(prediction, label)
                # for tensorboard
                self.writer.add_scalar(tag="ClassifierModelFromSDAE Validation Loss",
                                       scalar_value=classifier_model_loss.item(),
                                       global_step=epoch * len(dataloader) + i)
                # for visualization
                validation_classifier_epoch_data.append(prediction.detach())
                validation_labels.append(label)
                # for metric
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).cpu().numpy().sum()
        accuracy = f"Accuracy of validation epoch {epoch}: {round(correct / len(dataloader), 4)}"
        return accuracy, validation_classifier_epoch_data, validation_labels


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
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--log_path", type=str, default=LOG_DIR, help="Path of the log.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--classifier_lr", type=float, default=0.001, help="Learning rate of the Classifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--classifier_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of subprocesses for data loading.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="The steps of warm up.")
    args = parser.parse_args()

    train_dataset = CSFPDataset(args.train_input_file)
    validation_dataset = CSFPDataset(args.validation_input_file)
    train_dataloader, validation_dataloader = get_dataloader(train_dataset=train_dataset,
                                                             validation_dataset=validation_dataset,
                                                             collate_fn=custom_collate_fn,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.num_workers,
                                                             shuffle=True)
    train_total, validation_total = len(train_dataset), len(validation_dataset)
    train_input_size = next(iter(train_dataloader))["input_ids"].shape[1]
    validation_input_size = next(iter(validation_dataloader))["input_ids"].shape[1]
    sdae_model = StackedAutoEncoderModel(dimensions=[train_input_size, 2048, 1024, 512, 256, 128],
                                         final_activation=None).to(args.device)
    trainer = Trainer(args=args)
    print("Pretraining stage.")
    trainer.pretrain_ae_model(
        train_dataset,
        sdae_model,
        validation=validation_dataset,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
    )
    print("Training stage.")
    ae_optimizer = SGD(params=sdae_model.parameters(), lr=0.1, momentum=0.9)
    trainer.train_sdae_model(
        train_dataset,
        sdae_model,
        validation=validation_dataset,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=trainer.training_callback,
    )
    print("Classifier stage.")
    trainer.train_classifier(
        train_dataset,
        sdae_model,
        batch_size=args.batch_size,
        validation=validation_dataset
    )
    pass


