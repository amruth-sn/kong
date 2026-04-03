use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::bert::BertModel;

use crate::config::XdaConfig;

struct ClassifierHead {
    fc1: Linear,
    fc2: Linear,
}

impl ClassifierHead {
    fn load(vb: VarBuilder, hidden_size: usize, classifier_hidden: usize, num_labels: usize) -> Result<Self> {
        let fc1 = linear(hidden_size, classifier_hidden, vb.pp("classifier.0"))?;
        let fc2 = linear(classifier_hidden, num_labels, vb.pp("classifier.3"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(hidden_states)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

pub struct XdaModel {
    encoder: BertModel,
    classifier: ClassifierHead,
}

impl XdaModel {
    pub fn load(config: &XdaConfig, vb: VarBuilder) -> Result<Self> {
        let bert_config = config.to_bert_config();
        let encoder = BertModel::load(vb.pp("encoder"), &bert_config)?;
        let classifier = ClassifierHead::load(
            vb.clone(),
            config.hidden_size,
            config.classifier_hidden,
            config.num_labels,
        )?;
        Ok(Self { encoder, classifier })
    }

    /// inference
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let token_type_ids = input_ids.zeros_like()?;
        let hidden = self.encoder.forward(input_ids, &token_type_ids, Some(attention_mask))?;
        self.classifier.forward(&hidden)
    }
}