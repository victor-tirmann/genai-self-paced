import torch.nn as nn

from encoder import MultiHeadAttention, FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # masked multi-head attention -> prevents the decoder from attending to future tokens in the target sequence
        self_attention_output, self_attention_weights = self.masked_self_attention(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask,
        )

        #first LayerNorm -> adds residual connection (original embeddings) and normalizes the output of the masked self attention layer
        x = self.norm1(x + self.dropout(self_attention_output))

        #cross attention -> Query comes from the decoder (output of masked self attention), key and value come from the encoder output
        cross_attention_output, cross_attention_weights = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )

        #second LayerNorm -> adds residual connection (output of masked self attention) and normalizes the output of the cross attention layer
        x = self.norm2(x + self.dropout(cross_attention_output))

        ff_output = self.feed_forward(x)

        #third LayerNorm -> adds residual connection (output of cross attention) and normalizes the output of the feed forward layer
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attention_weights, cross_attention_weights