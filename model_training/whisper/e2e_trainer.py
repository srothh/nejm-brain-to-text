import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle
from collections import Counter

from model_training.dataset import BrainToTextDataset, train_test_split_indicies
from model_training.data_augmentations import gauss_smooth
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
import editdistance
from model_training.evaluate_model_helpers import remove_punctuation, _extract_transcription
from transformers.modeling_outputs import BaseModelOutput

from rnn_encoder import RNNEncoder

import torchaudio.functional as F  # for edit distance
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high')  # makes float32 matmuls faster on some GPUs
torch.backends.cudnn.deterministic = True  # makes training more reproducible
torch._dynamo.config.cache_size_limit = 64
# Silence some warnings about compilation, maybe investigate later
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from model_training.rnn_model import GRUDecoder, GRUEncoder


class End2EndModel_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder

    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.whisper = None

        self.best_val_PER = torch.inf  # track best PER for checkpointing
        self.best_val_loss = torch.inf  # track best loss for checkpointing

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=False)

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']:
            os.makedirs(self.args['checkpoint_dir'], exist_ok=False)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')

        if args['mode'] == 'train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'], 'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Configure device pytorch will use
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Set seed if provided
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize the model
        # ---- Initialize Whisper (decoder) ----
        
        whisper_name = self.args.get("whisper_model_name", "openai/whisper-medium")
        
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_name, language="English", task="transcribe")
        self.whisper_tokenizer = self.whisper_processor.tokenizer
        
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name).to(self.device)
        
        self.whisper.config.use_cache = False  # default is True 
        
        # Make generation deterministic about language/task
        # GC approach somehow does not work
        self.whisper.generation_config.language = "english"
        self.whisper.generation_config.task = "transcribe"
        gc = self.whisper.generation_config
        gc.language = None
        gc.task = None
        
        forced = self.whisper_processor.get_decoder_prompt_ids(language="english", task="transcribe")
        gc.forced_decoder_ids = forced  
        if self.args.get("freeze_whisper_decoder", True):
            for p in self.whisper.parameters():
                p.requires_grad = False

        # Try some unfreezing, only some cross attention for now
        unfreeze_layer_count = 0
        for name, param in self.whisper.named_parameters():
            if name.startswith("model.decoder.layers."):
                layer_id = int(name.split(".")[3])
                if layer_id >= (len(self.whisper.model.decoder.layers) - unfreeze_layer_count):
                    if (".encoder_attn." in name) or (".encoder_attn_layer_norm." in name):
                        param.requires_grad = True
        # Check this
        trainable = [(n, p.numel()) for n, p in self.whisper.named_parameters() if p.requires_grad]
        print("trainable whisper params:", len(trainable), "tensors",
      " | total:", sum(x[1] for x in trainable))
        self.trainable_whisper_params = [p for p in self.whisper.parameters() if p.requires_grad]
        # ---- Initialize baseline GRUDecoder ----
        from model_training.rnn_model import DayBiGRUEncoder
        
        d_model = int(self.whisper.config.d_model)
        
        def _clean_state_dict_keys(sd: dict) -> dict:
            out = {}
            for k, v in sd.items():
                k = k.replace("module.", "").replace("_orig_mod.", "")
                # allow either encoder.* or raw keys
                if k.startswith("encoder."):
                    k = k[len("encoder."):]
                out[k] = v
            return out
        
        pretrained_path = self.args.get(
            "pretrained_encoder_checkpoint",
            "model_training/trained_models/pretrained_rnn/checkpoint/best_checkpoint",
        )
        
        
        self.model = GRUEncoder(
            neural_dim=self.args["model"]["n_input_features"],      # 512
            n_units=self.args["model"]["n_units"],                  # 768
            n_days=len(self.args["dataset"]["sessions"]),
            n_classes=int(self.args["dataset"].get("n_classes", 41)),  # unused for enc output, but must exist
            rnn_dropout=self.args["model"]["rnn_dropout"],
            input_dropout=self.args["model"]["input_network"]["input_layer_dropout"],
            n_layers=self.args["model"]["n_layers"],
            patch_size=self.args["model"]["patch_size"],
            patch_stride=self.args["model"]["patch_stride"],
        
            head_type=self.args["model"].get("head_type", "none"),
            head_num_blocks=self.args["model"].get("head_num_blocks", 0),
            head_norm=self.args["model"].get("head_norm", "none"),
            head_dropout=self.args["model"].get("head_dropout", 0.0),
            head_activation=self.args["model"].get("head_activation", "gelu"),
        
            input_speckle_p=self.args["model"].get("input_speckle_p", 0.0),
            input_speckle_mode=self.args["model"].get("input_speckle_mode", "feature"),
        
            d_model=d_model,
        ).to(self.device)
        # Phoneme projection training experiment start
        self.n_phonemes = int(self.args.get("n_phonemes", 41))
        self.phoneme_head = nn.Linear(d_model, self.n_phonemes).to(self.device)
        nn.init.xavier_uniform_(self.phoneme_head.weight)
        # project back
        self.ctc_to_dmodel = nn.Linear(self.n_phonemes, d_model, bias=False).to(self.device)
        self.ctc_loss_weight = float(self.args.get("ctc_loss_weight", 0.0))     # auxiliary loss weight
        self.ctc_fuse_alpha  = float(self.args.get("ctc_fuse_alpha_initial",0.0))      # how much to add into enc
        self.detach_ctc_features = bool(self.args.get("detach_ctc_features", True))
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
        def _clean_state_dict_keys(sd: dict) -> dict:
            out = {}
            for k, v in sd.items():
                k = k.replace("module.", "").replace("_orig_mod.", "")
                out[k] = v
            return out

        pretrained_path = self.args.get(
            "pretrained_encoder_checkpoint",
            "model_training/trained_models/pretrained_rnn/checkpoint/best_checkpoint",
        )
        
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = _clean_state_dict_keys(sd)
        
        # If checkpoint is the WRAPPER, keys look like: encoder.* and phoneme_head.*
        if any(k.startswith("encoder.") for k in sd.keys()):
            enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            ph_sd  = {k[len("phoneme_head."):]: v for k, v in sd.items() if k.startswith("phoneme_head.")}
        else:
            # fallback: checkpoint might already be encoder-only
            enc_sd = sd
            ph_sd = None
        
        # Load encoder weights
        self.model.load_state_dict(enc_sd, strict=True)
        self.logger.info(f"Loaded pretrained encoder from: {pretrained_path}")
        
        # Load pretrained phoneme head weights (if present)
        if ph_sd is not None and len(ph_sd) > 0:
            self.phoneme_head.load_state_dict(ph_sd, strict=True)
            self.logger.info("Loaded pretrained phoneme_head weights too.")
        

        # experiment end
        #enc_state = {k.replace("encoder.", "", 1): v for k, v in phoneme_checkpoint.items() if k.startswith("encoder.")}
        #self.model.load_state_dict(enc_state, strict=True)

        # Maybe?
        if self.args.get("use_torch_compile", True):
            self.logger.info("Using torch.compile")
            self.model = torch.compile(self.model)

        self.model.to(self.device)

        self.logger.info(f"Initialized RNN decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()

        self.logger.info(
            f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Create datasets and dataloaders
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_train.hdf5') for s in
                            self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"], s, 'data_val.hdf5') for s in
                          self.args['dataset']['sessions']]

        # Ensure that there are no duplicate days
        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        # Split trials into train and test sets
        train_trials, _ = train_test_split_indicies(
            file_paths=train_file_paths,
            test_percentage=0,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=None,
        )
        _, val_trials = train_test_split_indicies(
            file_paths=val_file_paths,
            test_percentage=1,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=None,
        )

        # Save dictionaries to output directory to know which trials were train vs val
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f:
            json.dump({'train': train_trials, 'val': val_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None:
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')

        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies=train_trials,
            split='train',
            days_per_batch=self.args['dataset']['days_per_batch'],
            n_batches=self.args['num_training_batches'],
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # Dataset.__getitem__() already returns batches
            shuffle=self.args['dataset']['loader_shuffle'],
            num_workers=self.args['dataset']['num_dataloader_workers'],
            pin_memory=True
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split='test',
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,  # Dataset.__getitem__() already returns batches
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        self.logger.info("Successfully initialized datasets")

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args['lr_min'] / self.args['lr_max'],
                total_iters=self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)

        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")


        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Set rnn and/or input layers to not trainable if specified
        for name, param in self.model.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'gru' in name:
                param.requires_grad = False

            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

        # Send model to device
        self.model.to(self.device)

    def _make_whisper_decoder_inputs(self, sentences):
        tok = self.whisper_tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
    
        # Teacher forcing: decoder sees tokens up to t-1, predicts token t
        decoder_input_ids = tok.input_ids[:, :-1].contiguous()
        labels = tok.input_ids[:, 1:].clone().contiguous()
    
        # mask padding positions in labels (align with labels shape)
        labels[tok.attention_mask[:, 1:] == 0] = -100
    
        prompt_len = len(self.whisper_processor.get_decoder_prompt_ids())
        if prompt_len > 1:
            labels[:, :prompt_len - 1] = -100
    
        return tok, decoder_input_ids, labels

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        # named = list(self.model.named_parameters()) Non-phoneme fuse
    
        named = (
            list(self.model.named_parameters())
            + [(f"phoneme_head.{n}", p) for n, p in self.phoneme_head.named_parameters()]
            + [(f"ctc_to_dmodel.{n}", p) for n, p in self.ctc_to_dmodel.named_parameters()]
        )

        bias_params = [p for name, p in named if name.endswith("bias")]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        bias_ids = {id(p) for p in bias_params}
        day_ids = {id(p) for p in day_params}

        other_params = [p for name, p in named if (id(p) not in bias_ids) and (id(p) not in day_ids)]

        if len(day_params) != 0:
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias'},
                {'params': day_params, 'lr': self.args['lr_max_day'], 'weight_decay': self.args['weight_decay_day'],
                 'group_type': 'day_layer'},
                {'params': other_params, 'group_type': 'other'},
                {"params": self.trainable_whisper_params, 'group_type': 'whisper'}

            ]
        else:
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias'},
                {'params': other_params, 'group_type': 'other'},
                {"params": self.trainable_whisper_params, 'group_type': 'whisper'}
            ]

        optim = torch.optim.AdamW(
            param_groups,
            lr=self.args['lr_max'],
            betas=(self.args['beta0'], self.args['beta1']),
            eps=self.args['epsilon'],
            weight_decay=self.args['weight_decay'],
            fused=True
        )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day = self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay

            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)

            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio


        if len(optim.param_groups) == 4:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # biases
                lambda step: lr_lambda(
                    step,
                    lr_min_day / lr_max_day,
                    lr_decay_steps_day,
                    lr_warmup_steps_day,
                ),  # day params
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # rest of model weights
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # whisper layers, for now keep standard

            ]

        elif len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # biases
                lambda step: lr_lambda(
                    step,
                    lr_min_day / lr_max_day,
                    lr_decay_steps_day,
                    lr_warmup_steps_day,
                ),  # day params
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # biases
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps),  # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")

        return LambdaLR(optim, lr_lambdas, -1)

    def load_model_checkpoint(self, load_path):
        '''
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only=False)  # checkpoint is just a dict

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER']  # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf
        if "whisper_state_dict" in checkpoint:
            self.whisper.load_state_dict(checkpoint["whisper_state_dict"])
        
        if "phoneme_head_state_dict" in checkpoint:
            self.phoneme_head.load_state_dict(checkpoint["phoneme_head_state_dict"])
        
        if "ctc_to_dmodel_state_dict" in checkpoint:
            self.ctc_to_dmodel.load_state_dict(checkpoint["ctc_to_dmodel_state_dict"])

        self.model.to(self.device)

        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        '''
        Save a training checkpoint
        '''

        checkpoint = {
          "model_state_dict": self.model.state_dict(),
          "whisper_state_dict": self.whisper.state_dict(),
          "optimizer_state_dict": self.optimizer.state_dict(),
          "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
          "phoneme_head_state_dict": self.phoneme_head.state_dict(),
          "ctc_to_dmodel_state_dict": self.ctc_to_dmodel.state_dict(),
          "val_PER": PER,
          "val_loss": loss,
        }
        torch.save(checkpoint, save_path)

        self.logger.info("Saved model to checkpoint: " + save_path)

        # Save the args file alongside the checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def create_attention_mask(self, sequence_lengths):

        max_length = torch.max(sequence_lengths).item()

        batch_size = sequence_lengths.size(0)

        # Create a mask for valid key positions (columns)
        # Shape: [batch_size, max_length]
        key_mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length)
        key_mask = key_mask < sequence_lengths.unsqueeze(1)

        # Expand key_mask to [batch_size, 1, 1, max_length]
        # This will be broadcast across all query positions
        key_mask = key_mask.unsqueeze(1).unsqueeze(1)

        # Create the attention mask of shape [batch_size, 1, max_length, max_length]
        # by broadcasting key_mask across all query positions
        attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)

        # Convert boolean mask to float mask:
        # - True (valid key positions) -> 0.0 (no change to attention scores)
        # - False (padding key positions) -> -inf (will become 0 after softmax)
        attention_mask_float = torch.where(attention_mask,
                                           True,
                                           False)

        return attention_mask_float

    def transform_data(self, features, n_time_steps, mode='train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.eye(
                    channels,
                    device=features.device,
                    dtype=features.dtype,
                ).unsqueeze(0).expand(batch_size, -1, -1).clone()
            
                warp_mat = warp_mat + torch.randn_like(warp_mat) * self.transform_args['static_gain_std']
                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args[
                    'constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(
                    torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'],
                    dim=self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs=features,
                device=self.device,
                smooth_kernel_std=self.transform_args['smooth_kernel_std'],
                smooth_kernel_size=self.transform_args['smooth_kernel_size'],
            )

        return features, n_time_steps

    def train(self):
        '''
        Train the model
        '''

        # Set model to train mode (specificially to make sure dropout layers are engaged)
        self.model.train()
        # We are not training whisper
        self.whisper.eval()

        # create vars to track performance
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        # training params
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)

        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()

        # train for specified number of batches
        for i, batch in enumerate(self.train_loader):
                
            self.model.train()
            self.optimizer.zero_grad()

            # Train step
            start_time = time.time()

            # Move data to device
            features = batch['input_features'].to(self.device)
            phone_labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Use autocast for efficiency
            with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):

                # Apply augmentations to the data
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model'][
                    'patch_stride'] + 1).to(torch.int32)

                # Encode neural activity
                # If patching enabled, lengths must be the post-patching length (your adjusted_lens)
                patch_size = self.args["model"]["patch_size"]
                patch_stride = self.args["model"]["patch_stride"]
                
                if patch_size and patch_size > 0:
                    adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
                    adjusted_lens = torch.clamp(adjusted_lens, min=1)
                else:
                    adjusted_lens = n_time_steps.to(torch.int32)
                
                enc = self.model(features, day_indicies, lengths=adjusted_lens)
                phon_logits = self.phoneme_head(enc)              # (B, T, n_phonemes)
                phon_probs = phon_logits.softmax(dim=-1)
                if self.detach_ctc_features:
                    phon_probs = phon_probs.detach()
                enc = enc + (self.ctc_fuse_alpha * self.ctc_to_dmodel(phon_probs))

                B, T_enc, _ = enc.shape

                time_ids = torch.arange(T_enc, device=self.device).unsqueeze(0)
                mask_pad = time_ids >= adjusted_lens.unsqueeze(1)
                enc = enc.masked_fill(mask_pad.unsqueeze(-1), 0.0)
                enc_attn = (~mask_pad).long()
                encoder_outputs = BaseModelOutput(last_hidden_state=enc)
                # Some samples are too long currently for whisper, we need to clamp sadly. In the future, either downsampling or returning timestamps and splitting can be used
                MAX_SRC = 1500
                T = encoder_outputs.last_hidden_state.shape[1]
                if T > MAX_SRC:
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=encoder_outputs.last_hidden_state[:, :MAX_SRC]
                    )
                    enc_attn = enc_attn[:, :MAX_SRC]

                raw = batch.get('sentence_label', None)
                if raw is None:
                    raw = batch['transcriptions']
                if isinstance(raw, torch.Tensor):
                    raw = raw.cpu().numpy()
                sentences = []
                for s in raw:
                    if isinstance(s, (bytes, np.bytes_)):
                        sentences.append(s.decode("utf-8"))
                    else:
                        sentences.append(_extract_transcription(np.array(s)))

                tok, decoder_input_ids, whisper_labels = self._make_whisper_decoder_inputs(sentences)
                #DEBUG TODO: REMOVE
                if not hasattr(self, "_logged_whisper_label_example"):
                    self._logged_whisper_label_example = True
                
                    ex_ids = tok.input_ids[0].detach().cpu().tolist()
                    ex_mask = tok.attention_mask[0].detach().cpu().tolist()
                
                    # decode full padded sequence (for debugging)
                    decoded_full = self.whisper_tokenizer.decode(ex_ids, skip_special_tokens=False)
                
                    # decode only the non-pad part
                    n_valid = int(sum(ex_mask))
                    decoded_valid = self.whisper_tokenizer.decode(ex_ids[:n_valid], skip_special_tokens=False)
                
                    self.logger.info("[WHISPER LABEL CHECK] decoded_full:  " + repr(decoded_full))
                    self.logger.info("[WHISPER LABEL CHECK] decoded_valid: " + repr(decoded_valid))
                
                    # optional: also log the first few raw token ids to spot missing prefix tokens fast
                    self.logger.info("[WHISPER LABEL CHECK] first 16 token ids: " + str(ex_ids[:16]))

                out = self.whisper(
                    encoder_outputs=encoder_outputs,
                    attention_mask=enc_attn,
                    decoder_input_ids=decoder_input_ids,

                    labels=whisper_labels,
                )
                # loss = out.loss Non-phoneme
                whisper_loss = out.loss
                ctc_loss = self.ctc_loss(
                    log_probs=phon_logits.float().log_softmax(dim=-1).transpose(0, 1),  # (T,B,C)
                    targets=phone_labels,
                    input_lengths=adjusted_lens,
                    target_lengths=phone_seq_lens,
                )
                whisper_loss_item = float(whisper_loss.detach().item())
                ctc_loss_item = float(ctc_loss.detach().item())

                loss = whisper_loss + (self.ctc_loss_weight * ctc_loss)
                

            loss.backward()

            # Clip gradient
            if self.args['grad_norm_clip_value'] > 0:
                # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                #                                           max_norm=self.args['grad_norm_clip_value'],
                #                                           error_if_nonfinite=True,
                #                                           foreach=True
                #                                           )
                to_clip = (
                    list(self.model.parameters())
                    + list(self.phoneme_head.parameters())
                    + list(self.ctc_to_dmodel.parameters())
                    + list(self.trainable_whisper_params)  # empty if all frozen, fine
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    to_clip,
                    max_norm=self.args["grad_norm_clip_value"],
                    error_if_nonfinite=True,
                    foreach=True,
                )

            self.optimizer.step()
            self.learning_rate_scheduler.step()

            # Save training metrics
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Incrementally log training progress
            if i % self.args['batches_per_train_log'] == 0:
                if i % self.args['batches_per_train_log'] == 0:
                    self.logger.info(
                        f"Train batch {i}: "
                        f"loss: {loss.detach().item():.4f} "
                        f"whisper: {whisper_loss_item:.4f} "
                        f"ctc: {ctc_loss_item:.4f} "
                        f"(ctc*w={self.ctc_loss_weight:.3f} -> {self.ctc_loss_weight*ctc_loss_item:.4f}) "
                        f"grad norm: {grad_norm:.2f} "
                        f"time: {train_step_duration:.3f}"
                    )

            # Incrementally run a test step
            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")

                # Calculate metrics on val data
                start_time = time.time()
                val_metrics = self.validation(loader=self.val_loader, return_logits=self.args['save_val_logits'],
                                              return_data=self.args['save_val_data'])
                val_step_duration = time.time() - start_time

                # Log info
                self.logger.info(f'Val batch {i}: ' +
                                 f'WER (avg): {val_metrics["avg_PER"]:.4f} ' +
                                 f'Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                                 f'time: {val_step_duration:.3f}')

                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_PERs'].keys():
                        self.logger.info(
                            f"{self.args['dataset']['sessions'][day]} val WER: {val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}")

                # Save metrics
                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                # Determine if new best day. Based on if PER is lower, or in the case of a PER tie, if loss is lower
                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test WER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss):
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:

                    # Checkpoint if metrics have improved
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER,
                                                   self.best_val_loss)

                    # save validation metrics to pickle file
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0

                else:
                    val_steps_since_improvement += 1

                # Optionally save this validation checkpoint, regardless of performance
                if self.args['save_all_val_steps']:
                    self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}',
                                               val_metrics['avg_PER'])

                # Early stopping
                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(
                        f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break

        # Log final training steps
        training_duration = time.time() - train_start_time

        self.logger.info(f'Best avg val WER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model
        if self.args['save_final_model']:
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results

        return train_stats

    def validation(self, loader, return_logits=False, return_data=False):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()
        self.whisper.eval()
        total_edit_distance = 0
        total_true_length = 0
        losses = []

        day_per = {}
        examples_to_print = int(self.args.get("val_print_examples", 5))
        printed = 0

        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1:
                day_per[d] = {'total_edit_distance': 0, 'total_seq_length': 0}

        for i, batch in enumerate(loader):
            features = batch['input_features'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)
            phone_labels = batch['seq_class_ids'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)

            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0:
                if self.args.get('log_val_skip_logs', False):
                    self.logger.info(f"Skipping validation on day {day}")
                continue

            with torch.no_grad():
                with torch.autocast(
                        device_type="cuda",
                        enabled=(self.args['use_amp'] and self.device.type == "cuda"),
                        dtype=torch.bfloat16
                ):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')
                    adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model'][
                        'patch_stride'] + 1).to(torch.int32)

                    # If patching enabled, lengths must be the post-patching length (your adjusted_lens)
                    patch_size = self.args["model"]["patch_size"]
                    patch_stride = self.args["model"]["patch_stride"]
                    
                    if patch_size and patch_size > 0:
                        adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
                        adjusted_lens = torch.clamp(adjusted_lens, min=1)
                    else:
                        adjusted_lens = n_time_steps.to(torch.int32)
                    
                    enc = self.model(features, day_indicies, lengths=adjusted_lens)
                    # Phoneme experiment start
                    phon_logits = self.phoneme_head(enc)
                    phon_probs = phon_logits.softmax(dim=-1)
                    if self.detach_ctc_features:
                        phon_probs = phon_probs.detach()
                    
                    enc = enc + (self.ctc_fuse_alpha * self.ctc_to_dmodel(phon_probs))
                    # Phoneme experiment end

                    B, T_enc, _ = enc.shape
                    if not hasattr(self, "_len_debug_done"):
                        self._len_debug_done = True
                        self.logger.info(
                            f"[LEN DEBUG] n_time_steps[0]={int(n_time_steps[0])} "
                            f"adjusted_lens[0]={int(adjusted_lens[0])} "
                            f"T_enc={enc.shape[1]}"
                        )

                    time_ids = torch.arange(T_enc, device=self.device).unsqueeze(0)
                    mask_pad = time_ids >= adjusted_lens.unsqueeze(1)
                    enc = enc.masked_fill(mask_pad.unsqueeze(-1), 0.0)

                    assert enc.shape[-1] == self.whisper.config.d_model

                    enc_attn = (~mask_pad).long()
                    assert (enc_attn.sum(dim=1) > 0).all()

                    encoder_outputs = BaseModelOutput(last_hidden_state=enc)

                    # Prefer sentence_label if present; fallback to transcriptions
                    raw = batch.get('sentence_label', None)
                    if raw is None:
                        raw = batch['transcriptions']

                    if isinstance(raw, torch.Tensor):
                        raw = raw.cpu().numpy()

                    sentences = []
                    for s in raw:
                        if isinstance(s, (bytes, np.bytes_)):
                            sentences.append(s.decode("utf-8"))
                        else:
                            sentences.append(_extract_transcription(np.array(s)))

                    tok, decoder_input_ids, whisper_labels = self._make_whisper_decoder_inputs(sentences)

                    out = self.whisper(
                        encoder_outputs=encoder_outputs,
                        attention_mask=enc_attn,
                        decoder_input_ids=decoder_input_ids,
                        labels=whisper_labels,
                    )
                    # loss = out.loss Non-phoneme
                    whisper_loss = out.loss
                    ctc_loss = self.ctc_loss(
                        log_probs=phon_logits.float().log_softmax(dim=-1).transpose(0, 1),  # (T, B, C)
                        targets=phone_labels,
                        input_lengths=adjusted_lens,
                        target_lengths=phone_seq_lens,
                    )
                    loss = whisper_loss + (self.ctc_loss_weight * ctc_loss)
                    losses.append(float(loss.item()))
                with torch.autocast(
                    device_type="cuda",
                    enabled=self.args["use_amp"],
                    dtype=torch.bfloat16,
                ):
                    MAX_SRC = 1500  # Whisper's limit (as implemented in generate)
                    T = encoder_outputs.last_hidden_state.shape[1]
                    if T > MAX_SRC:
                        encoder_outputs = BaseModelOutput(
                            last_hidden_state=encoder_outputs.last_hidden_state[:, :MAX_SRC]
                        )
                        enc_attn = enc_attn[:, :MAX_SRC]


                    gen_ids = self.whisper.generate(
                        encoder_outputs=encoder_outputs,
                        attention_mask=enc_attn,
                        forced_decoder_ids=self.whisper.generation_config.forced_decoder_ids,
                        max_new_tokens=self.args.get("max_new_tokens", 64),
                        num_beams=self.args.get("num_beams", 5),
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1,
                        length_penalty=0.0,
                        eos_token_id=self.whisper_tokenizer.eos_token_id,
                        pad_token_id=self.whisper_tokenizer.eos_token_id,
                        
                    )
                hyps = self.whisper_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                block_nums = batch.get("block_nums", None)
                trial_nums = batch.get("trial_nums", None)
                if isinstance(block_nums, torch.Tensor):
                    block_nums = block_nums.detach().cpu().tolist()
                if isinstance(trial_nums, torch.Tensor):
                    trial_nums = trial_nums.detach().cpu().tolist()
                
                for j, (ref, hyp) in enumerate(zip(sentences, hyps)):
                    if printed >= examples_to_print:
                        break
                
                    ref_clean = remove_punctuation(ref).strip()
                    hyp_clean = remove_punctuation(hyp).strip()
                
                    ref_words = ref_clean.split()
                    hyp_words = hyp_clean.split()
                    ed = editdistance.eval(ref_words, hyp_words)
                    wer = (ed / len(ref_words)) if len(ref_words) > 0 else float("inf")
                
                    bnum = block_nums[j] if block_nums is not None and j < len(block_nums) else "?"
                    tnum = trial_nums[j] if trial_nums is not None and j < len(trial_nums) else "?"
                
                    self.logger.info(
                        f"[VAL EX {printed+1}] day={day} block={bnum} trial={tnum}\n"
                        f"  REF: {ref_clean}\n"
                        f"  HYP: {hyp_clean}\n"
                        f"  WER: {wer:.3f}  (ed={ed}, n_ref={len(ref_words)})"
                    )
                    printed += 1

                batch_ed = 0
                batch_words = 0
                for ref, hyp in zip(sentences, hyps):
                    ref_clean = remove_punctuation(ref).strip()
                    hyp_clean = remove_punctuation(hyp).strip()
                    ed = editdistance.eval(ref_clean.split(), hyp_clean.split())
                    batch_ed += ed
                    batch_words += len(ref_clean.split())

                day_per[day]['total_edit_distance'] += batch_ed
                day_per[day]['total_seq_length'] += batch_words

                total_edit_distance += batch_ed
                total_true_length += batch_words

        avg_loss = float(np.mean(losses)) if len(losses) else float("inf")
        avg_WER = (total_edit_distance / total_true_length) if total_true_length > 0 else 0.0

        return {
            'day_PERs': day_per,  # keep key name so baseline logging still works
            'avg_PER': float(avg_WER),  # keep key name so baseline checkpoint logic still works
            'avg_loss': avg_loss,
        }