use half::bf16;

pub struct RopeScalingConfig<'a> {
    factor: bf16,
    high_freq_factor: bf16,
    low_freq_factor: bf16,
    original_max_position_embeddings: u32,
    rope_type: &'a str,
}

pub struct Config<'a> {
    attention_bias: bool,
    attention_dropout: bf16,
    bos_token_id: u32,
    eos_token_id: u32,
    head_dim: u32,
    hidden_act: &'a str,
    hidden_size: u32,
    initializer_range: bf16,
    intermediate_size: u32,
    max_position_embeddings: u32,
    mlp_bias: bool,
    model_type: &'a str,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    pretraining_tp: u32,
    rms_norm_eps: bf16,
    rope_scaling: RopeScalingConfig<'a>,
    rope_theta: bf16,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: u32,
}

// TODO load this from config.json?
pub fn load_config() -> Config<'static> {
    Config {
        attention_bias: false,
        attention_dropout: bf16::from_f32(0.0),
        bos_token_id: 128000,
        eos_token_id: 128001,
        head_dim: 64,
        hidden_act: "silu",
        hidden_size: 2048,
        initializer_range: bf16::from_f32(0.02),
        intermediate_size: 8192,
        max_position_embeddings: 131072,
        mlp_bias: false,
        model_type: "llama",
        num_attention_heads: 32,
        num_hidden_layers: 16,
        num_key_value_heads: 8,
        pretraining_tp: 1,
        rms_norm_eps: bf16::from_f32(1e-05),
        rope_scaling: RopeScalingConfig {
            factor: bf16::from_f32(32.0),
            high_freq_factor: bf16::from_f32(4.0),
            low_freq_factor: bf16::from_f32(1.0),
            original_max_position_embeddings: 8192,
            rope_type: "llama3",
        },
        rope_theta: bf16::from_f32(500000.0),
        tie_word_embeddings: true,
        use_cache: true,
        vocab_size: 128256,
    }
}
