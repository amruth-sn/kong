import torch
from src.model import XdaModel, XdaConfig


def test_model_config_defaults():
    config = XdaConfig()
    assert config.vocab_size == 261
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.intermediate_size == 3072
    assert config.num_labels == 3
    assert config.max_position_embeddings == 514  # 512 + [CLS] + [SEP]


def test_model_forward_shape():
    config = XdaConfig(
        num_hidden_layers=2,  # smaller for test speed
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
    )
    model = XdaModel(config)

    batch_size = 4
    seq_len = 514
    input_ids = torch.randint(0, 261, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(input_ids, attention_mask)
    assert logits.shape == (batch_size, seq_len, 3)


def test_model_parameter_count():
    """Sanity check that full-size model has ~87M params."""
    config = XdaConfig()
    model = XdaModel(config)
    total = sum(p.numel() for p in model.parameters())
    # Allow some tolerance — should be in the 80-95M range
    assert 80_000_000 < total < 100_000_000, f"Param count {total} outside expected range"