pub struct ByteTokenizer {
    pub pad_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub unk_id: u32,
    pub vocab_size: u32,
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self {
            pad_id: 256,
            cls_id: 257,
            sep_id: 258,
            mask_id: 259,
            unk_id: 260,
            vocab_size: 261,
        }
    }
}

impl ByteTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// If `max_length` is less than the length of the encoded bytes, truncate the encoded bytes to `max_length`.
    /// Otherwise, pad the encoded bytes to `max_length` with the pad token.
    pub fn encode(&self, bytes: &[u8], max_length: Option<usize>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(bytes.len() + 2);
        ids.push(self.cls_id);
        ids.extend(bytes.iter().map(|&b| b as u32));
        ids.push(self.sep_id);

        if let Some(max_len) = max_length {
            ids.truncate(max_len);
            ids.resize(max_len, self.pad_id);
        }

        ids
    }

    pub fn attention_mask(&self, token_ids: &[u32]) -> Vec<u32> {
        token_ids
            .iter()
            .map(|&id| if id == self.pad_id { 0 } else { 1 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_token_ids() {
        let tok = ByteTokenizer::new();
        assert_eq!(tok.pad_id, 256);
        assert_eq!(tok.cls_id, 257);
        assert_eq!(tok.sep_id, 258);
        assert_eq!(tok.vocab_size, 261);
    }

    #[test]
    fn test_encode_bytes() {
        let tok = ByteTokenizer::new();
        let raw = [0x55u8, 0x48, 0x89, 0xe5];
        let ids = tok.encode(&raw, None);
        assert_eq!(ids, vec![257, 0x55, 0x48, 0x89, 0xe5, 258]);
    }

    #[test]
    fn test_encode_with_padding() {
        let tok = ByteTokenizer::new();
        let raw = [0x55u8, 0x48];
        let ids = tok.encode(&raw, Some(8));
        assert_eq!(ids, vec![257, 0x55, 0x48, 258, 256, 256, 256, 256]);
    }

    #[test]
    fn test_attention_mask() {
        let tok = ByteTokenizer::new();
        let raw = [0x55u8, 0x48];
        let ids = tok.encode(&raw, Some(8));
        let mask = tok.attention_mask(&ids);
        assert_eq!(mask, vec![1, 1, 1, 1, 0, 0, 0, 0]);
    }
}