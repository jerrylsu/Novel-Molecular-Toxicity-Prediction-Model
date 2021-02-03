import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List


def build_units(
    dimensions: Iterable[int], activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear
    layer followed by an activation layer.
    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """

    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension in sliding_window(2, dimensions)
    ]


def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    """
    Default function to initialise the weights in a the Linear units of the StackedDenoisingAutoEncoder.
    :param weight: weight Tensor of the Linear unit
    :param bias: bias Tensor of the Linear unit
    :param gain: gain for use in initialiser
    :return: None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)


class AutoencoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: int,
        activation: Optional[torch.nn.Module] = nn.ReLU(),
        gain: float = nn.init.calculate_gain("relu"),
        corruption: Optional[torch.nn.Module] = None,
        tied: bool = False,
    ) -> None:
        """
        Autoencoder composed of two Linear units with optional encoder activation and corruption.
        :param embedding_dimension: embedding dimension, input to the encoder
        :param hidden_dimension: hidden dimension, output of the encoder
        :param activation: optional activation unit, defaults to nn.ReLU()
        :param gain: gain for use in weight initialisation
        :param corruption: optional unit to apply to corrupt input during training, defaults to None
        :param tied: whether the autoencoder weights are tied, defaults to False
        """
        super(AutoencoderLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.activation = activation
        self.gain = gain
        self.corruption = corruption
        # encoder parameters
        self.encoder_weight = Parameter(
            torch.Tensor(hidden_dimension, embedding_dimension)
        )
        self.encoder_bias = Parameter(torch.Tensor(hidden_dimension))
        self._initialise_weight_bias(self.encoder_weight, self.encoder_bias, self.gain)
        # decoder parameters
        self._decoder_weight = (
            Parameter(torch.Tensor(embedding_dimension, hidden_dimension))
            if not tied
            else None
        )
        self.decoder_bias = Parameter(torch.Tensor(embedding_dimension))
        self._initialise_weight_bias(self._decoder_weight, self.decoder_bias, self.gain)

    @property
    def decoder_weight(self):
        return (
            self._decoder_weight
            if self._decoder_weight is not None
            else self.encoder_weight.t()
        )

    @staticmethod
    def _initialise_weight_bias(weight: torch.Tensor, bias: torch.Tensor, gain: float):
        """
        Initialise the weights in a the Linear layers of the DenoisingAutoencoder.
        :param weight: weight Tensor of the Linear layer
        :param bias: bias Tensor of the Linear layer
        :param gain: gain for use in initialiser
        :return: None
        """
        if weight is not None:
            nn.init.xavier_uniform_(weight, gain)
        nn.init.constant_(bias, 0)

    def copy_weights(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
        """
        Utility method to copy the weights of self into the given encoder and decoder, where
        encoder and decoder should be instances of torch.nn.Linear.
        :param encoder: encoder Linear unit
        :param decoder: decoder Linear unit
        :return: None
        """
        encoder.weight.data.copy_(self.encoder_weight)
        encoder.bias.data.copy_(self.encoder_bias)
        decoder.weight.data.copy_(self.decoder_weight)
        decoder.bias.data.copy_(self.decoder_bias)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        transformed = F.linear(batch, self.encoder_weight, self.encoder_bias)
        if self.activation is not None:
            transformed = self.activation(transformed)
        if self.corruption is not None:
            transformed = self.corruption(transformed)
        return transformed

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        return F.linear(batch, self.decoder_weight, self.decoder_bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))


class StackedAutoEncoderModel(nn.Module):
    def __init__(
        self,
        dimensions: List[int],
        activation: torch.nn.Module = nn.ReLU(),
        final_activation: Optional[torch.nn.Module] = nn.ReLU(),
        weight_init: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
    ):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimensions input is the list of dimensions occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].
        :param dimensions: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        :param weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        :param gain: gain parameter to pass to weight_init
        """
        super(StackedAutoEncoderModel, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        # construct the encoder
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(
            build_units([self.dimensions[-2], self.dimensions[-1]], None)
        )
        self.encoder = nn.Sequential(*encoder_units)
        # construct the decoder
        decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(
            build_units([self.dimensions[1], self.dimensions[0]], final_activation)
        )
        self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.
        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)
