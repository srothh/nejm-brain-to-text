#!/usr/bin/env python3
import os
import time
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import editdistance
from omegaconf import OmegaConf

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

# --- match baseline import behavior / make model_training importable ---
ROOT = Path(__file__).resolve().parents[2]
MT   = ROOT / "model_training"
WSP  = MT / "whisper"
sys.path[:0] = [str(ROOT), str(MT), str(WSP)]

# --- USE YOUR HELPERS (exact file you pasted) ---
from model_training.evaluate_model_helpers import (
    load_h5py_file,
    remove_punctuation,
    gauss_smooth,   # imported from helper module namespace (it imports gauss_smooth)
)

# encoder used in your trainer
from model_training.rnn_model import DayBiGRUEncoder, GRUEncoder


# argument parser for command line arguments (baseline style)
parser = argparse.ArgumentParser(description='Evaluate encoder+Whisper model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='model_training/whisper/trained_models/baseline_rnn',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='data/hdf5_data_final',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--csv_path', type=str, default='data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata about the dataset (relative to the current working directory).')
parser.add_argument('--gpu_number', type=int, default=0,
                    help='GPU number to use for model inference. Set to -1 to use CPU.')
args = parser.parse_args()

# paths to model and data directories
model_path = args.model_path
data_dir = args.data_dir

# define evaluation type
eval_type = args.eval_type  # can be 'val' or 'test'

if data_dir.startswith("/") and not os.path.exists(data_dir):
    data_dir = data_dir[1:]  # strip exactly one leading slash
if model_path.startswith("/") and not os.path.exists(model_path):
    model_path = model_path[1:]

# load csv file
b2txt_csv_df = pd.read_csv(args.csv_path)

# load model args (baseline style)
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# set up gpu device (baseline style)
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# -------------------------
# Initialize Whisper + Encoder (replaces GRUDecoder)
# -------------------------
whisper_name = model_args.get("whisper_model_name", "openai/whisper-medium")
lang = model_args.get("whisper_language", "en")
task = model_args.get("whisper_task", "transcribe")

whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name).to(device)
whisper.config.use_cache = False

forced = whisper_processor.get_decoder_prompt_ids(language=lang, task=task)
whisper.generation_config.forced_decoder_ids = forced
whisper.eval()

d_model = int(whisper.config.d_model)
model = GRUEncoder(
    neural_dim=model_args["model"]["n_input_features"],      # 512
    n_units=model_args["model"]["n_units"],                  # 768
    n_days=len(model_args["dataset"]["sessions"]),
    n_classes=int(model_args["dataset"].get("n_classes", 41)),  # unused for enc output, but must exist
    rnn_dropout=model_args["model"]["rnn_dropout"],
    input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
    n_layers=model_args["model"]["n_layers"],
    patch_size=model_args["model"]["patch_size"],
    patch_stride=model_args["model"]["patch_stride"],

    head_type=model_args["model"].get("head_type", "none"),
    head_num_blocks=model_args["model"].get("head_num_blocks", 0),
    head_norm=model_args["model"].get("head_norm", "none"),
    head_dropout=model_args["model"].get("head_dropout", 0.0),
    head_activation=model_args["model"].get("head_activation", "gelu"),

    input_speckle_p=model_args["model"].get("input_speckle_p", 0.0),
    input_speckle_mode=model_args["model"].get("input_speckle_mode", "feature"),

    d_model=d_model,
).to(device)
model.eval()
import torch.nn as nn

n_phonemes = int(model_args.get("n_phonemes", 41))
ctc_fuse_alpha = float(model_args.get("ctc_fuse_alpha", model_args.get("ctc_fuse_alpha_initial", 0.0)))

phoneme_head = nn.Linear(d_model, n_phonemes).to(device).eval()
ctc_to_dmodel = nn.Linear(n_phonemes, d_model, bias=False).to(device).eval()

# load model weights (baseline-style: load best_checkpoint and strip prefixes)
checkpoint_path = os.path.join(model_path, 'checkpoint/best_checkpoint')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
if "phoneme_head_state_dict" in checkpoint:
    phoneme_head.load_state_dict(checkpoint["phoneme_head_state_dict"], strict=True)
    print("Loaded phoneme_head from checkpoint")

if "ctc_to_dmodel_state_dict" in checkpoint:
    ctc_to_dmodel.load_state_dict(checkpoint["ctc_to_dmodel_state_dict"], strict=True)
    print("Loaded ctc_to_dmodel from checkpoint")

state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
if "whisper_state_dict" in checkpoint:
    whisper.load_state_dict(checkpoint["whisper_state_dict"], strict=True)
    print("Loaded Whisper weights from checkpoint")
else:
    print("No whisper_state_dict in checkpoint; using pretrained Whisper")

new_state = {}
for k, v in state.items():
    k2 = k.replace("module.", "").replace("_orig_mod.", "")
    new_state[k2] = v
model.load_state_dict(new_state)

# -------------------------
# load data for each session (baseline style)
# -------------------------
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')

        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()

# -------------------------
# "Predicting phoneme sequences" stage (baseline structure)
# Here: run gauss_smooth + encoder + whisper.generate, store per-trial outputs
# -------------------------
patch_size = int(model_args['model'].get('patch_size', 0) or 0)
patch_stride = int(model_args['model'].get('patch_stride', 1) or 1)
MAX_SRC = int(model_args.get("whisper_max_src", 1500))
max_new_tokens = int(model_args.get("max_new_tokens", 64))
num_beams = int(model_args.get("num_beams", 1))

with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        data['logits'] = []       # keep same key as baseline, but we store generated ids here
        data['pred_sentence'] = []  # store decoded sentence for convenience

        input_layer = model_args['dataset']['sessions'].index(session)

        for trial in range(len(data['neural_features'])):
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)

            # baseline uses bfloat16 on CUDA
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # baseline-style autocast usage + baseline gauss_smooth call (padding='valid')
            with torch.autocast(device_type="cuda", enabled=model_args.get('use_amp', True), dtype=torch.bfloat16):
                if model_args['dataset']['data_transforms'].get('smooth_data', True):
            
                    neural_input = gauss_smooth(
                        inputs=neural_input,
                        device=device,
                        smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                        smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                        padding='valid',
                    )

                # compute lengths AFTER smoothing, then AFTER patching (encoder expects post-patch length)
                T_smooth = int(neural_input.shape[1])
                if patch_size > 0:
                    adj_len = (T_smooth - patch_size) // patch_stride + 1
                    adj_len = max(adj_len, 1)
                else:
                    adj_len = max(T_smooth, 1)

                lengths = torch.tensor([adj_len], device=device, dtype=torch.long)

                with torch.no_grad():
                    enc = model(
                        x=neural_input,
                        day_idx=torch.tensor([input_layer], device=device, dtype=torch.long),
                        lengths=lengths,
                    )
                    phon_logits = phoneme_head(enc)                 # (B, T, n_phonemes)
                    phon_probs = phon_logits.softmax(dim=-1)        
                    enc = enc + (ctc_fuse_alpha * ctc_to_dmodel(phon_probs))


                    _, T_enc, _ = enc.shape
                    time_ids = torch.arange(T_enc, device=device).unsqueeze(0)
                    mask_pad = time_ids >= lengths.unsqueeze(1)
                    enc = enc.masked_fill(mask_pad.unsqueeze(-1), 0.0)
                    attn = (~mask_pad).long()

                    if T_enc > MAX_SRC:
                        enc = enc[:, :MAX_SRC]
                        attn = attn[:, :MAX_SRC]

                    encoder_outputs = BaseModelOutput(last_hidden_state=enc.float())

                    gen_ids = whisper.generate( encoder_outputs=encoder_outputs,
                                                attention_mask=attn, max_new_tokens=64,
                                                num_beams=5, length_penalty=0.0,
                                                eos_token_id=whisper_processor.tokenizer.eos_token_id,
                                                pad_token_id=whisper_processor.tokenizer.eos_token_id, )
            pred = whisper_processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

            data['logits'].append(gen_ids.detach().cpu().numpy())   # keep baseline key name
            data['pred_sentence'].append(pred)

            pbar.update(1)
pbar.close()

# -------------------------
# "Running remote language model" stage (baseline structure)
# Here: just take stored pred_sentence as best_candidate_sentence
# -------------------------
lm_results = {
    'session': [],
    'block': [],
    'trial': [],
    'true_sentence': [],
    'pred_sentence': [],
}

with tqdm(total=total_test_trials, desc='Running remote language model', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            best_candidate_sentence = test_data[session]['pred_sentence'][trial]

            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            if eval_type == 'val':
                # EXACTLY like baseline: store sentence_label as-is from loader
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
            lm_results['pred_sentence'].append(best_candidate_sentence)

            pbar.update(1)
pbar.close()

# -------------------------
# WER computation (kept EXACTLY like baseline)
# -------------------------
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0

    lm_results['edit_distance'] = []
    lm_results['num_words'] = []

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i]).strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        lm_results['edit_distance'].append(ed)
        lm_results['num_words'].append(len(true_sentence.split()))

        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f'True sentence:       {true_sentence}')
        print(f'Predicted sentence:  {pred_sentence}')
        print(f'WER: {ed} / {100 * len(true_sentence.split())} = {ed / len(true_sentence.split()):.2f}%')
        print()

    print(f'Total true sentence length: {total_true_length}')
    print(f'Total edit distance: {total_edit_distance}')
    print(f'Aggregate Word Error Rate (WER): {100 * total_edit_distance / total_true_length:.2f}%')

# -------------------------
# write predicted sentences to CSV (baseline structure)
# -------------------------
output_file = os.path.join(
    model_path,
    f'encoder_whisper_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.csv'
)
ids = [i for i in range(len(lm_results['pred_sentence']))]
df_out = pd.DataFrame({'id': ids, 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
print(f"\nWrote predictions to: {output_file}")