from dataclasses import dataclass

import torch
from torch import nn
from transformers import BertConfig, BertModel


@dataclass
class XdaConfig:
    vocab_size: int = 261                # 256 bytes + 5 special tokens
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 514   # 512 byte window + [CLS] + [SEP]
    num_labels: int = 3                  # non-function, function_start, function_body
    classifier_hidden: int = 256         # MLP hidden dimension
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    def to_bert_config(self) -> BertConfig:
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        )


class XdaModel(nn.Module):
    """XDA: BERT encoder with a 2-layer MLP token classification head."""

    def __init__(self, config: XdaConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BertModel(config.to_bert_config(), add_pooling_layer=False)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.classifier_hidden, config.num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits of shape (batch, seq_len, num_labels)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state)