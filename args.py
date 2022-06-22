from argparse import ArgumentParser


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--d_model', type=int, default=256
    )
    group.add_argument(
        '--h', type=int, default=4
    )
    group.add_argument(
        '--add_lnorm', type=bool, default=True
    )
    group.add_argument(
        '--hidden_size', type=int, default=256
    )
    group.add_argument(
        '--p_dopout', type=float, default=0.1
    )
    group.add_argument(
        '--add_lnorm', type=bool, default=True
    )
    group.add_argument(
        '--left_shift', type=int, default=-1,
        help='The number below the center in the sliding attention'
        )
    group.add_argument(
        '--right_shift', type=int, default=4,
        help='The number beyond the center in the sliding attention'
        )
    group.add_argument(
        '--max_steps', type=int, default=4,
        help="""
        The maximum number of steps the sliding attention allowed to take
        """
        )
    group.add_argument(
        '--att_bandwidth', type=int, default=50,
        help="""
        The maximum number of steps the sliding attention allowed to take
        """
        )
    group.add_argument(
        '--ldc_lambda', type=float, default=0.01
        )
    group.add_argument(
        '--stop_weight', type=float, default=8
        )
    group.add_argument(
        '--bottleneck_size', type=int, default=30
        )
    group.add_argument(
        '--n_layers', type=int, default=4,
        help='Number of transformer layers'
        )
    group.add_argument(
        '--spk_emb_size', type=int, default=128
    )


def add_training_args(parser):
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--train_path', type=str
    )
    group.add_argument(
        '--test_path', type=str
    )
    group.add_argument(
        '--checkpoint_dir', type=str
    )
    group.add_argument(
        '--pretrained_path', type=str, required=False, default=None
    )
    group.add_argument(
        '--steps_per_ckpt', type=int
    )
    group.add_argument(
        '--epochs', type=int
    )
    group.add_argument(
        '--batch_size', type=int
    )
    group.add_argument(
        '--opt_eps', type=float, default=1e-9
    )
    group.add_argument(
        '--opt_beta1', type=float, default=0.9
    )
    group.add_argument(
        '--opt_beta2', type=float, default=0.98
    )
    group.add_argument(
        '--opt_warmup_staps', type=int, default=4000
    )


def add_data_args(parser):
    group = parser.add_argument_group('Data')
    group.add_argument(
        '--tokenizer_path', type=str, default=None
    )
    group.add_argument(
        '--n_mels', type=int, default=80
    )
    group.add_argument(
        '--sampling_rate', type=int, default=16000
    )
    group.add_argument(
        '--hop_size', type=int, default=200
    )
    group.add_argument(
        '--window_size', type=int, default=800
    )
    group.add_argument(
        '--n_fft', type=int, default=1024
    )
    group.add_argument(
        '--sep', type=str, default='|'
    )


def get_argparse():
    parser = ArgumentParser()
    add_model_args(parser)
    add_training_args(parser)
    add_data_args(parser)
    return parser


def get_args() -> dict:
    parser = get_argparse()
    args = parser.parse_args()
    return args


def get_model_args(
        args: dict, vocab_size: int, pad_idx: int, n_speakers: int
        ) -> dict:
    d_model = args['d_model']
    h = args['h']
    p_dopout = args['p_dopout']
    model_args = {
        'n_layers': args['n_layers'],
        'device': args['device']
    }
    pos_emb_key = 'pos_emb_params'
    encoder_key = 'encoder_params'
    decoder_key = 'decoder_params'
    speaker_mod_key = 'speaker_mod_params'
    prenet_key = 'prenet_params'
    pred_key = 'pred_params'
    pos_emb_params = {
        pos_emb_key: dict()
    }
    encoder_params = {
        encoder_key: dict()
    }
    decoder_params = {
        decoder_key: dict()
    }
    speaker_mod_params = {
        speaker_mod_key: dict()
    }
    prenet_params = {
        prenet_key: dict()
    }
    pred_params = {
        pred_key: dict()
    }
    pos_emb_params[pos_emb_key] = {
        'd_model': d_model,
        'vocab_size': vocab_size,
        'pad_idx': pad_idx,
        'device': args['device'],
        'add_lnorm': args['add_lnorm']
    }
    encoder_params[encoder_key] = {
        'd_model': d_model,
        'h': h,
        'hidden_size': args['hidden_size'],
        'p_dopout': p_dopout
    }
    decoder_params[decoder_key] = {
        'd_model': d_model,
        'h': h,
        'p_dropout': p_dopout,
        'left_shift': args['left_shift'],
        'right_shift': args['right_shift'],
        'max_steps': args['max_steps'],
        'hidden_size': args['hidden_size']
    }
    speaker_mod_params[speaker_mod_key] = {
        'n_speakers': n_speakers,
        'emb_size': args['spk_emb_size'],
        'd_model': d_model
    }
    prenet_params[prenet_key] = {
        'inp_size': args['n_mels'],
        'bottleneck_size': args['bottleneck_size'],
        'd_model': d_model,
        'p_dropout': p_dopout
    }
    pred_params[pred_key] = {
        'd_model': d_model,
        'n_mels': args['n_mels']
    }
    return {
        **model_args,
        **pos_emb_params,
        **encoder_params,
        **decoder_params,
        **speaker_mod_params,
        **prenet_params,
        **pred_params
    }


def get_loss_args(args: dict) -> dict:
    return {
        'h': args['h'],
        'dc_strength': args['ldc_lambda'],
        'dc_bandwidth': args['att_bandwidth'],
        'stop_weight': args['stop_weight']
    }


def get_optim_args(args: dict) -> dict:
    return {
        'betas': (args['opt_beta1'], args['opt_beta2']),
        'eps': args['opt_eps'],
        'warmup_staps': args['warmup_staps'],
        'd_model': args['d_model']
    }


def get_aud_args(args: dict) -> dict:
    return {
        'sampling_rate': args['sampling_rate'],
        'win_size': args['window_size'],
        'hop_size': args['hop_size'],
        'n_mels': args['n_mels'],
        'n_fft': args['n_fft']
    }


def get_data_args(args: dict) -> dict:
    return {
        'sep': args['sep'],
        'batch_size': args['batch_size']
    }


def get_trainer_args(args: dict) -> dict:
    return {
        'save_dir': args['checkpoint_dir'],
        'steps_per_ckpt': args['steps_per_ckpt'],
        'epochs': args['epochs'],
        'device': args['device']
    }
