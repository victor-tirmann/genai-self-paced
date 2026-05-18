import math
import torch
import torch.nn as nn

from encoder import PositionalEncoding, EncoderBlock
from decoder import DecoderBlock


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 256,
        max_sequence_len: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        # Paddings to ensure the sequences in a batch have the same length. The model should ignore these when processing the input.
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len = max_sequence_len)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        #final linear layer to project the decoder output to the target vocabulary size
        # for each token position, one score per vocabulary token
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        #creates a mask for the source sequence, where positions with padding tokens are masked out (False) and non-padding tokens are not masked (True)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        """
        Combines:
        1. padding mask
        2. causal mask
        Causal mask prevents the decoder from seeing future target tokens.
        """

        batch_size, tgt_len = tgt.shape

        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        #lower triangular matrix of shape (tgt_len, tgt_len) 
        #where the diagonal and below are True (allowed to attend) and above the diagonal are False (not allowed to attend)
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()

        return padding_mask & causal_mask

    def encode(self, src):
        src_mask = self.make_src_mask(src)

        x = self.src_embedding(src)
        x = x * math.sqrt(self.d_model)

        x = self.positional_encoding(x)
        x = self.dropout(x)

        encoder_attention_maps = []

        for layer in self.encoder_layers:
            x, attention_weights = layer(x, src_mask)
            encoder_attention_maps.append(attention_weights)

        return x, src_mask, encoder_attention_maps

    def decode(self, tgt, encoder_output, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)

        x = self.tgt_embedding(tgt)
        x = x * math.sqrt(self.d_model)

        x = self.positional_encoding(x)
        x = self.dropout(x)

        decoder_self_attention_maps = []
        decoder_cross_attention_maps = []

        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(
                x,
                encoder_output,
                src_mask,
                tgt_mask,
            )

            decoder_self_attention_maps.append(self_attn)
            decoder_cross_attention_maps.append(cross_attn)

        return x, decoder_self_attention_maps, decoder_cross_attention_maps

    def forward(self, src, tgt):
        encoder_output, src_mask, encoder_attn = self.encode(src)

        decoder_output, decoder_self_attn, decoder_cross_attn = self.decode(
            tgt,
            encoder_output,
            src_mask,
        )

        logits = self.output_linear(decoder_output)

        attention_maps = {
            "encoder_self_attention": encoder_attn,
            "decoder_self_attention": decoder_self_attn,
            "decoder_cross_attention": decoder_cross_attn,
        }

        return logits, attention_maps