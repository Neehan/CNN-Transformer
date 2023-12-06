import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

from models.Transformer import Model as VanillaTransformer


class Model(nn.Module):
    """
    A CNN + Vanilla Transformer Model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        # self.enc_in = configs.enc_in # encoder sequence length

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),  # output 4 channel x enc_in
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(int(configs.enc_in // 2) * 4, configs.enc_in),
        )

        self.transformer = VanillaTransformer(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        print(x_enc.shape)
        batch_size, num_vars, length = x_enc.shape
        x_enc = self.cnn(x_enc.view(batch_size * num_vars, 1, length))
        x_enc = x_enc.view(batch_size, num_vars, -1)
        return self.transformer(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        return self.transformer.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def anomaly_detection(self, x_enc):
        return self.transformer.anomaly_detection(x_enc)

    def classification(self, x_enc, x_mark_enc):
        return self.transformer.classification(x_enc, x_mark_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            return self.transformer.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            return self.transformer(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
