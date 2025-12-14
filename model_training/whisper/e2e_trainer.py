import torch
from torch.utils.data import DataLoader
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

from model_training.rnn_model import GRUDecoder


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
        lang = self.args.get("whisper_language", "en")
        task = self.args.get("whisper_task", "transcribe")
        
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_name)
        self.whisper_tokenizer.set_prefix_tokens(language=lang, task=task)
        
        self.whisper.generation_config.language = lang
        self.whisper.generation_config.task = task

        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name).to(self.device)

        if self.args.get("freeze_whisper_decoder", True):
            for p in self.whisper.model.decoder.parameters():
                p.requires_grad = False

        self.forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
            language=self.args.get("whisper_language", "en"),
            task=self.args.get("whisper_task", "transcribe"),
        )

        # ---- Initialize baseline GRUDecoder exactly as before ----
        gru_model = GRUDecoder(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_days=len(self.args['dataset']['sessions']),
            n_classes=self.args['dataset']['n_classes'],
            rnn_dropout=self.args['model']['rnn_dropout'],
            input_dropout=self.args['model']['input_network']['input_layer_dropout'],
            n_layers=self.args['model']['n_layers'],
            patch_size=self.args['model']['patch_size'],
            patch_stride=self.args['model']['patch_stride'],
        )

        # ---- Wrap it with projection Encoder ----
        proj_in_dim = self.args.get("rnn_proj_in_dim", self.args["dataset"]["n_classes"])
        d_model = int(self.whisper.config.d_model)
        self.model = RNNEncoder(gru_model=gru_model, proj_in_dim=proj_in_dim, d_model=d_model)

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

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        named = list(self.model.named_parameters())

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
                {'params': other_params, 'group_type': 'other'}
            ]
        else:
            param_groups = [
                {'params': bias_params, 'weight_decay': 0, 'group_type': 'bias'},
                {'params': other_params, 'group_type': 'other'}
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

        if len(optim.param_groups) == 3:
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_PER': PER,
            'val_loss': loss
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
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim=0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

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
            labels = batch['seq_class_ids'].to(self.device)
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
                enc = self.model(features, day_indicies)
                B, T_enc, _ = enc.shape

                time_ids = torch.arange(T_enc, device=self.device).unsqueeze(0)
                mask_pad = time_ids >= adjusted_lens.unsqueeze(1)
                enc = enc.masked_fill(mask_pad.unsqueeze(-1), 0.0)
                enc_attn = (~mask_pad).long()
                encoder_outputs = BaseModelOutput(last_hidden_state=enc)

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

                tok = self.whisper_tokenizer(
                    sentences, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                labels = tok.input_ids.clone()
                labels[tok.attention_mask == 0] = -100

                out = self.whisper(
                    encoder_outputs=encoder_outputs,
                    attention_mask=enc_attn,
                    labels=labels,
                )
                loss = out.loss

            loss.backward()

            # Clip gradient
            if self.args['grad_norm_clip_value'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           max_norm=self.args['grad_norm_clip_value'],
                                                           error_if_nonfinite=True,
                                                           foreach=True
                                                           )

            self.optimizer.step()
            self.learning_rate_scheduler.step()

            # Save training metrics
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Incrementally log training progress
            if i % self.args['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: ' +
                                 f'loss: {(loss.detach().item()):.2f} ' +
                                 f'grad norm: {grad_norm:.2f} '
                                 f'time: {train_step_duration:.3f}')

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

                    enc = self.model(features, day_indicies)  # (B, T_enc, d_model)
                    B, T_enc, _ = enc.shape

                    time_ids = torch.arange(T_enc, device=self.device).unsqueeze(0)
                    mask_pad = time_ids >= adjusted_lens.unsqueeze(1)
                    enc = enc.masked_fill(mask_pad.unsqueeze(-1), 0.0)
                    enc_attn = (~mask_pad).long()

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

                    tok = self.whisper_tokenizer(
                        sentences, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)

                    labels = tok.input_ids.clone()
                    # Blank weight
                    labels[tok.attention_mask == 0] = -100

                    out = self.whisper(
                        encoder_outputs=encoder_outputs,
                        attention_mask=enc_attn,
                        labels=labels,
                    )
                    loss = out.loss
                    losses.append(float(loss.item()))
                with torch.autocast(
                    device_type="cuda",
                    enabled=self.args["use_amp"],
                    dtype=torch.bfloat16,
                ):

                    gen_ids = self.whisper.generate(
                        encoder_outputs=encoder_outputs,
                        attention_mask=enc_attn,
                        max_new_tokens=self.args.get("max_new_tokens", 64),
                        num_beams=self.args.get("num_beams", 1),
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
