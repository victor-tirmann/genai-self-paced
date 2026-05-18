import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from transformer import Transformer


class Tokenizer:
    def __init__(self):
        #mapping for encoding - text to numbers
        self.token_to_id = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,

            "i": 3,
            "like": 4,
            "cats": 5,
            "dogs": 6,
            "you": 7,
            "love": 8,

            "mie": 9,
            "imi": 10,
            "plac": 11,
            "pisicile": 12,
            "cainii": 13,
            "tu": 14,
            "iubesti": 15,
        }

        #reverse mapping for decoding 0: "<pad>", 1: "<bos>", etc. - numbers back to text
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text, add_bos=False, add_eos=True):
        #simple tokenization -> individual words
        tokens = text.lower().split()

        ids = []

        if add_bos:
            ids.append(self.token_to_id["<bos>"])

    #convert words to token ids
        for token in tokens:
            ids.append(self.token_to_id[token])

        #TRUE for encoder, FALSE for decoder
        if add_eos:
            ids.append(self.token_to_id["<eos>"])

        return ids

    def decode(self, ids):
        tokens = []

        #convert token ids back to words, skipping special tokens like <pad>, <bos>, <eos>
        for idx in ids:
            token = self.id_to_token[int(idx)]

            if token in ["<pad>", "<bos>", "<eos>"]:
                continue

            tokens.append(token)

        return " ".join(tokens)


# Adds padding <pad> to sequences to ensure they have the same length within a batch. The model should ignore these padding tokens during processing.
def pad_sequence(seq, max_len, pad_idx=0):
    return seq + [pad_idx] * (max_len - len(seq))


def create_toy_batch(tokenizer):
    #src - encoder input
    #tgt - decoder target
    pairs = [
        ("i like cats", "mie imi plac pisicile"),
        ("i like dogs", "mie imi plac cainii"),
        ("you love cats", "tu iubesti pisicile"),
        ("you love dogs", "tu iubesti cainii"),
    ]

    src_sequences = []
    tgt_sequences = []

    # convert text pairs to token id sequences using the tokenizer, adding <bos> and <eos> tokens as needed for the decoder input and target
    # add_bos=False because encoder does not need a BOS token
    # add_eos=True so encoder knows where sentence ends
    for src_text, tgt_text in pairs:
        src_ids = tokenizer.encode(src_text, add_bos=False, add_eos=True)
        tgt_ids = tokenizer.encode(tgt_text, add_bos=True, add_eos=True)

        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)

    #find longest sequence in the batch to pad all sequences to the same length
    max_src_len = max(len(seq) for seq in src_sequences)
    max_tgt_len = max(len(seq) for seq in tgt_sequences)

    src_batch = torch.tensor([
        pad_sequence(seq, max_src_len) for seq in src_sequences
    ])

    tgt_batch = torch.tensor([
        pad_sequence(seq, max_tgt_len) for seq in tgt_sequences
    ])

    return src_batch, tgt_batch


def train_toy_model():
    tokenizer = Tokenizer()

    vocab_size = len(tokenizer.token_to_id)
    pad_idx = tokenizer.token_to_id["<pad>"]

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
        pad_idx=pad_idx,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    src_batch, tgt_batch = create_toy_batch(tokenizer)

    model.train()

    for epoch in range(300):
        tgt_input = tgt_batch[:, :-1]
        tgt_expected = tgt_batch[:, 1:]

        logits, _ = model(src_batch, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_expected.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

    return model, tokenizer


def translate(model, tokenizer, text, max_len=10):
    model.eval()

    bos_idx = tokenizer.token_to_id["<bos>"]
    eos_idx = tokenizer.token_to_id["<eos>"]

    src_ids = tokenizer.encode(text, add_bos=False, add_eos=True)
    src = torch.tensor([src_ids])

    encoder_output, src_mask, _ = model.encode(src)

    generated = torch.tensor([[bos_idx]])

    for _ in range(max_len):
        decoder_output, _, _ = model.decode(
            generated,
            encoder_output,
            src_mask,
        )

        logits = model.output_linear(decoder_output)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return tokenizer.decode(generated[0])


def visualize_attention(attention_tensor, x_tokens, y_tokens, title, head=0):
    attention = attention_tensor[0, head].detach().cpu().numpy()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        attention,
        xticklabels=x_tokens,
        yticklabels=y_tokens,
        cmap="viridis",
        annot=True,
        fmt=".2f",
    )

    plt.xlabel("Tokens attended to")
    plt.ylabel("Tokens attending")
    plt.title(f"{title} - Head {head}")
    plt.tight_layout()
    plt.show()


def inspect_attention(model, tokenizer, src_text, tgt_text, layer_index=0, head=0):
    model.eval()

    src_ids = tokenizer.encode(src_text, add_bos=False, add_eos=True)
    tgt_ids = tokenizer.encode(tgt_text, add_bos=True, add_eos=True)

    src = torch.tensor([src_ids])

    # During training/inference, decoder receives target shifted right.
    tgt = torch.tensor([tgt_ids[:-1]])

    src_tokens = [tokenizer.id_to_token[i] for i in src_ids]
    tgt_tokens = [tokenizer.id_to_token[i] for i in tgt_ids[:-1]]

    with torch.no_grad():
        _, attention_maps = model(src, tgt)

    encoder_attn = attention_maps["encoder_self_attention"][layer_index]

    visualize_attention(
        encoder_attn,
        x_tokens=src_tokens,
        y_tokens=src_tokens,
        title="Encoder Self-Attention",
        head=head,
    )

    decoder_self_attn = attention_maps["decoder_self_attention"][layer_index]

    visualize_attention(
        decoder_self_attn,
        x_tokens=tgt_tokens,
        y_tokens=tgt_tokens,
        title="Decoder Masked Self-Attention",
        head=head,
    )

    decoder_cross_attn = attention_maps["decoder_cross_attention"][layer_index]

    visualize_attention(
        decoder_cross_attn,
        x_tokens=src_tokens,
        y_tokens=tgt_tokens,
        title="Decoder Cross-Attention",
        head=head,
    )


def compare_attention_behavior(model, tokenizer):
    examples = [
        ("i like cats", "mie imi plac pisicile"),
        ("i like dogs", "mie imi plac cainii"),
        ("you love cats", "tu iubesti pisicile"),
        ("you love dogs", "tu iubesti cainii"),
    ]

    for src_text, tgt_text in examples:
        print()
        print(f"Source: {src_text}")
        print(f"Expected target: {tgt_text}")
        print(f"Model translation: {translate(model, tokenizer, src_text)}")

        inspect_attention(
            model,
            tokenizer,
            src_text=src_text,
            tgt_text=tgt_text,
            layer_index=0,
            head=0,
        )


if __name__ == "__main__":
    model, tokenizer = train_toy_model()

    print()
    print("Translations:")
    print("i like cats ->", translate(model, tokenizer, "i like cats"))
    print("i like dogs ->", translate(model, tokenizer, "i like dogs"))
    print("you love cats ->", translate(model, tokenizer, "you love cats"))
    print("you love dogs ->", translate(model, tokenizer, "you love dogs"))

    print()
    print("Inspecting one example:")
    inspect_attention(
        model,
        tokenizer,
        src_text="i like cats",
        tgt_text="mie imi plac pisicile",
        layer_index=0,
        head=0,
    )