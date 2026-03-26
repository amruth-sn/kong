use candle_core::{Device, IndexOp, Result, Tensor};
use kong_types::function::{AmbiguousRegion, DetectionSource, FunctionBoundary};

use crate::model::XdaModel;
use crate::tokenizer::ByteTokenizer;

pub struct XdaDetector {
    model: XdaModel,
    tokenizer: ByteTokenizer,
    device: Device,
    confidence_threshold: f32,
    window_size: usize,
    stride: usize,
}

/// Per-byte prediction from a single window.
#[derive(Debug, Clone)]
struct BytePrediction {
    address: u64,
    label: u8,       // 0=non-func, 1=func_start, 2=func_body
    confidence: f32,
}

impl XdaDetector {
    pub fn new(
        model: XdaModel,
        device: Device,
        confidence_threshold: f32,
    ) -> Self {
        Self {
            model,
            tokenizer: ByteTokenizer::new(),
            device,
            confidence_threshold,
            window_size: 512,
            stride: 256,
        }
    }

    /// Predict function boundaries in ambiguous regions.
    pub fn predict(&self, regions: &[AmbiguousRegion]) -> Result<Vec<FunctionBoundary>> {
        let mut boundaries = Vec::new();

        for region in regions {
            let region_predictions = self.predict_region(region)?;
            boundaries.extend(region_predictions);
        }

        Ok(boundaries)
    }

    fn predict_region(&self, region: &AmbiguousRegion) -> Result<Vec<FunctionBoundary>> {
        let bytes = &region.bytes;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        let windows = self.create_windows(bytes);
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        let predictions = self.run_batch(&windows)?;

        // Merge overlapping window predictions by AVERAGING confidence scores
        let merged = self.merge_predictions(&predictions, bytes.len(), region.start);

        Ok(self.extract_boundaries(&merged))
    }

    /// since binary code can contain 0x00, we track how many bytes are real vs padding.
    fn create_windows(&self, bytes: &[u8]) -> Vec<(Vec<u8>, usize)> {
        let mut windows = Vec::new();
        let mut offset = 0;

        while offset < bytes.len() {
            let end = (offset + self.window_size).min(bytes.len());
            let actual_len = end - offset;
            let mut window = bytes[offset..end].to_vec();

            if window.len() < self.window_size {
                window.resize(self.window_size, 0);
            }

            windows.push((window, actual_len));
            offset += self.stride;

            if end == bytes.len() {
                break;
            }
        }

        windows
    }

    fn run_batch(&self, windows: &[(Vec<u8>, usize)]) -> Result<Vec<Vec<(u8, f32)>>> {
        let seq_len = self.window_size + 2; // +2 for [CLS] and [SEP]
        let batch_size = windows.len();

        let mut all_ids = Vec::with_capacity(batch_size * seq_len);
        let mut all_mask = Vec::with_capacity(batch_size * seq_len);

        for (window, _actual_len) in windows {
            let ids = self.tokenizer.encode(window, Some(seq_len));
            let mask = self.tokenizer.attention_mask(&ids);
            all_ids.extend(ids.iter().map(|&x| x as i64));
            all_mask.extend(mask.iter().map(|&x| x as i64));
        }

        let input_ids = Tensor::from_vec(all_ids, (batch_size, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(all_mask, (batch_size, seq_len), &self.device)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;

        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

        let mut batch_predictions = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let window_probs = probs.i((i, 1..seq_len - 1))?; // skip [CLS] at 0, [SEP] at end
            let window_probs = window_probs.to_vec2::<f32>()?;

            let actual_len = windows[i].1;
            let mut preds = Vec::with_capacity(actual_len);

            for j in 0..actual_len {
                let p = &window_probs[j];
                let (label, confidence) = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, &conf)| (idx as u8, conf))
                    .unwrap_or((0, 0.0));
                preds.push((label, confidence));
            }

            batch_predictions.push(preds);
        }

        Ok(batch_predictions)
    }

    fn merge_predictions(
        &self,
        batch_predictions: &[Vec<(u8, f32)>],
        total_bytes: usize,
        base_address: u64,
    ) -> Vec<BytePrediction> {
        let mut votes: Vec<([f32; 3], u32)> = vec![([0.0; 3], 0); total_bytes];

        for (window_idx, preds) in batch_predictions.iter().enumerate() {
            let offset = window_idx * self.stride;
            for (byte_idx, &(label, confidence)) in preds.iter().enumerate() {
                let global_idx = offset + byte_idx;
                if global_idx < total_bytes {
                    votes[global_idx].0[label as usize] += confidence;
                    votes[global_idx].1 += 1;
                }
            }
        }

        votes
            .into_iter()
            .enumerate()
            .map(|(idx, (scores, count))| {
                let count = count.max(1) as f32;
                let avg_scores = [
                    scores[0] / count,
                    scores[1] / count,
                    scores[2] / count,
                ];
                let (label, confidence) = avg_scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(l, &c)| (l as u8, c))
                    .unwrap_or((0, 0.0));

                BytePrediction {
                    address: base_address + idx as u64,
                    label,
                    confidence,
                }
            })  
            .collect()
    }

    fn extract_boundaries(&self, predictions: &[BytePrediction]) -> Vec<FunctionBoundary> {
        let mut boundaries = Vec::new();
        let mut func_start: Option<&BytePrediction> = None;

        for pred in predictions {
            match pred.label {
                1 if pred.confidence >= self.confidence_threshold => {
                    if let Some(start) = func_start.take() {
                        boundaries.push(FunctionBoundary {
                            start: start.address,
                            end: pred.address,
                            confidence: start.confidence,
                            source: DetectionSource::Ml,
                        });
                    }
                    func_start = Some(pred);
                }
                0 if func_start.is_some() => {
                    // Non-function byte after function body, in this case we close the function
                    let start = func_start.take().unwrap();
                    boundaries.push(FunctionBoundary {
                        start: start.address,
                        end: pred.address,
                        confidence: start.confidence,
                        source: DetectionSource::Ml,
                    });
                }
                _ => {}
            }
        }

        if let Some(start) = func_start {
            if let Some(last) = predictions.last() {
                boundaries.push(FunctionBoundary {
                    start: start.address,
                    end: last.address + 1,
                    confidence: start.confidence,
                    source: DetectionSource::Ml,
                });
            }
        }

        boundaries
    }
}