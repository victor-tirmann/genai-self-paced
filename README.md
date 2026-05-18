# Minimal Transformer Exploration

Code was generated using agents and online resources using the Attention Is All You Need as example.
The goal was to roughly see the implementation steps and understand the steps since I could not find a pre-trained model that was small enough to run locally and expose meaningful attention scores to actually visualize them.

The model includes:

- Source and target embeddings
- Sinusoidal positional encoding
- Encoder self-attention
- Decoder masked self-attention
- Decoder cross-attention
- Feed-forward layers (FFN)
- Residual connections
- Layer normalization
- Final linear projection to vocabulary logits

The model was trained on a very small toy translation dataset, for example:

```text
i like cats → mie imi plac pisicile
```

## Because the model and dataset are very small, the learned attention patterns are not always stable between runs.

## Attention Map Observations

I attached the results of one run of the model in the `examples` folder.

### Decoder Cross-Attention

In the decoder cross-attention map , target tokens attend to source tokens.

For example:

- The token `plac` strongly attends to `cats`
- The token `pisicile` attends mostly to `<eos>` (kind of makes sense, in both English and Romanian cats / pisicile is the final word in the sequence) and partly to `cats`

---

### Decoder Masked Self-Attention

In the decoder masked self-attention map, attention is mostly diagonal.

This means:

- Each target token mainly attends to itself (and only themselves in this run)
- Future tokens are blocked by the causal mask (and somehow past tokens as well in this run)

---

### Encoder Self-Attention

In the encoder self-attention map, source tokens attend to other source tokens.

The patterns are sometimes unintuitive, such as:

```text
like → <eos>
```

This is expected because:

- The model is extremely small
- The dataset contains only a few examples
- Training is very limited

With larger datasets and deeper models, attention patterns usually become more meaningful and distributed across multiple tokens.

---

## Modifying Inputs and Observing Attention Behavior

To observe attention behavior, the same visualization can be run with modified inputs such as:

```text
i like cats → mie imi plac pisicile
i like dogs → mie imi plac cainii
```

The expected behavior is:

- Cross-attention shifts from source token `cats` toward `dogs`
- Cross-attention shifts from target token `pisicile` toward `cainii`

This demonstrates that attention dynamically adapts based on the input sequence, but then again, this is a small model.

---

## Context Length, Latency, and Inference Cost

Attention maps grow with sequence length.

If the sequence contains `n` tokens, self-attention creates an:

```text
n × n
```

attention matrix per head, per layer.

This has important implications:

- Longer context increases memory usage
- Longer context increases computation cost
- Latency increases because more attention scores must be computed
- Inference becomes more expensive as sequence length grows
