use serde::{Deserialize, Serialize};
use std::path::Path;
use std::error::Error;
use std::fs::read_to_string;
use serde_json;
use candle_transformers::models::bert::Config as BertConfig;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XdaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_labels: usize,
    pub classifier_hidden: usize,
}

impl Default for XdaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 261,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 514,
            num_labels: 3,
            classifier_hidden: 256,
        }
    }
}

impl XdaConfig {
    pub fn from_json(path: &Path) -> Result<Self, Box<dyn Error>> {
        let content = read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn to_bert_config(&self) -> BertConfig {
        BertConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            hidden_dropout_prob: 0.0,
            ..Default::default()
        }
    }
}