local train_subset = "file_dir/train.json";
local valid_subset = "file_dir/valid.json";
local global_cmvn = "file_dir/global_cmvn";
local vocabulary_path = "file_dir/vocabulary";
local serialization_dir = "file_dir/serialization_dir";

{
    "variable": {
        "vocab_size": 4233,
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

        "train_subset": train_subset,
        "valid_subset": valid_subset,
        "global_cmvn": global_cmvn,
        "vocabulary_path": vocabulary_path,

        "padding_idx": -1,
        "ctc_weight": 0.3,
        "smoothing": 0.1,

        "batch_size": 16,
    },
    "dataset_reader": {
        "type": "speech_to_text_json",
        "tokenizer": {
            "type": "cjk_bpe_tokenizer"
        }
    },
    "train_data_path": $.variable.train_subset,
    "validation_data_path": $.variable.valid_subset,
    "vocabulary": {
        "directory_path": $.variable.vocabulary_path
    },
    "collate_fn": {
        "type": "preprocess_cfn",
        "ignore_id": -1,
        "preprocess_list": [
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
                "vocabulary": {
                    "directory_path": $.variable.vocabulary_path
                }
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
        "namespace": "tokens"
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
    "optimizer": {
        "type": "bert_adam",
        "lr": 2e-3,
        "warmup": 0.1,
        "t_total": 530000,
        "schedule": "warmup_linear"
    },
    "trainer": {
        "type": "pytorch_lightning",
        "patience": 10,
        "validation_metric": "+val_acc_att",
        "num_workers": 8,
        "max_epochs": 240,
        "min_epochs": 200,
        "batch_size": $.variable.batch_size,
        "accumulate_grad_batches": 4,
        "serialization_dir": serialization_dir,
        "num_serialized_models_to_keep": 20,
        "cuda_device": [0],
        "gradient_clip_val": 5.0,
        "log_every_n_steps": 10
    },
    "predictor": {
        "type": "asr_predictor",
    }
}
