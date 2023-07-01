local global_cmvn = "file_dir/global_cmvn";
local symbol_table_file = "file_dir/dict.txt";

{
    "variable": {
        "vocab_size": 29,
        "sample_rate": 8000,

        "feat_dim": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,

        "encoder_hidden_size": 256,
        "encoder_attention_heads": 4,
        "encoder_linear_units": 2048,
        "encoder_num_layers": 6,

        "decoder_attention_heads": 4,
        "decoder_linear_units": 2048,
        "decoder_num_layers": 6,

        "global_cmvn": global_cmvn,
        "symbol_table_file": symbol_table_file,

        "padding_idx": -1,
        "ctc_weight": 0.3,
        "smoothing": 0.1,

        "batch_size": 16,

    },

    "preprocess": [
        {
            "type": "load_wav"
        },
        {
            "type": "resample",
            "resample_rate": $.variable.sample_rate
        },
        {
            "type": "speed_perturb"
        },
        {
            "type": "cjk_bpe_tokenize"
        },
        {
            "type": "map_tokens_to_ids",
            "symbol_table_file": $.variable.symbol_table_file
        },
        {
            "type": "waveform_to_fbank",
            "num_mel_bins": $.variable.feat_dim,
            "frame_length": $.variable.frame_length,
            "frame_shift": $.variable.frame_shift,
            "dither": $.variable.dither
        },
        {
            "type": "spectrum_aug",
            "num_t_mask": 2,
            "num_f_mask": 2,
            "max_t": 50,
            "max_f": 10
        }
    ],

    "train_dataset_reader": {
        "type": "speech_to_text_json",
        "tokenizer": {
            "type": "cjk_bpe_tokenizer"
        }
    },
    "valid_dataset_reader": {
        "type": "speech_to_text_json",
        "tokenizer": {
            "type": "cjk_bpe_tokenizer"
        }
    },
    "model": {
        "type": "hybrid_ctc_attention_asr_model",
        "vocab_size": $.variable.vocab_size,
        "encoder": {
            "type": "conformer_encoder",
            "subsampling": {
                "type": "subsampling4",
                "input_dim": $.variable.feat_dim,
                "output_dim": $.variable.encoder_hidden_size,
                "dropout_rate": 0.1,
                "positional_encoding": {
                    "type": "sinusoidal",
                    "embedding_dim": $.variable.encoder_hidden_size,
                    "dropout_rate": 0.1
                }
            },
            "global_cmvn": {
                "cmvn_file": $.variable.global_cmvn,
                "is_json_cmvn": true
            },
            "output_size": $.variable.encoder_hidden_size,
            "attention_heads": $.variable.encoder_attention_heads,
            "linear_units": $.variable.encoder_linear_units,
            "num_blocks": $.variable.encoder_num_layers
        },
        "decoder": {
            "type": "transformer_decoder",
            "vocab_size": $.variable.vocab_size,
            "input_size": $.variable.encoder_hidden_size,
            "attention_heads": $.variable.decoder_attention_heads,
            "linear_units": $.variable.decoder_linear_units,
            "num_blocks": $.variable.decoder_num_layers
        },
        "ctc_loss": {
            "vocab_size": $.variable.vocab_size,
            "encoder_output_size": $.variable.encoder_hidden_size
        },
        "ctc_weight": $.variable.ctc_weight,
        "att_loss": {
            "vocab_size": $.variable.vocab_size,
            "padding_idx": $.variable.padding_idx,
            "smoothing": $.variable.smoothing
        }
    },
}
