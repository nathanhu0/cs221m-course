"""Pre-compute neuron activations for the SAE notebook in shards.

Run this once to generate sharded data files. Students don't need to run this.
Each shard contains ~1M tokens of activations.
"""
import os
import torch
from tqdm.auto import tqdm
from nnsight import LanguageModel
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = LanguageModel("EleutherAI/pythia-70m-v0", device_map=device, dispatch=True)
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

import random

print("Loading dataset...")
dataset = load_dataset("Skylion007/openwebtext", split="train[:100000]")
texts = [t for t in dataset["text"] if len(t) > 100]
random.seed(42)
random.shuffle(texts)
print(f"Using {len(texts)} texts (shuffled)")

LAYER_IDX = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
TOKENS_PER_SHARD = 1_000_000

OUTPUT_DIR = "activations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

shard_activations = []
shard_input_ids = []
shard_seq_lengths = []
shard_token_count = 0
shard_idx = 0

for batch_start in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Layer {LAYER_IDX}"):
    batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

    tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        with model.trace(input_ids):
            hidden = model.gpt_neox.layers[LAYER_IDX].mlp.dense_4h_to_h.input.save()

    for batch_idx in range(input_ids.shape[0]):
        mask = attention_mask[batch_idx].bool()
        ids = input_ids[batch_idx][mask].cpu()
        acts = hidden[batch_idx][mask].cpu().half()
        shard_input_ids.append(ids)
        shard_activations.append(acts)
        shard_seq_lengths.append(len(ids))
        shard_token_count += len(ids)

        # Write shard when we hit the target size
        if shard_token_count >= TOKENS_PER_SHARD:
            output_path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:03d}.pt")
            torch.save({
                "layer_idx": LAYER_IDX,
                "model": "EleutherAI/pythia-70m-v0",
                "activations": torch.cat(shard_activations, dim=0),
                "input_ids": torch.cat(shard_input_ids, dim=0),
                "seq_lengths": shard_seq_lengths,
            }, output_path)
            size_gb = os.path.getsize(output_path) / 1e9
            print(f"\nSaved shard {shard_idx}: {shard_token_count} tokens ({size_gb:.2f} GB)")

            shard_activations = []
            shard_input_ids = []
            shard_seq_lengths = []
            shard_token_count = 0
            shard_idx += 1

# Save final partial shard
if shard_token_count > 0:
    output_path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:03d}.pt")
    torch.save({
        "layer_idx": LAYER_IDX,
        "model": "EleutherAI/pythia-70m-v0",
        "activations": torch.cat(shard_activations, dim=0),
        "input_ids": torch.cat(shard_input_ids, dim=0),
        "seq_lengths": shard_seq_lengths,
    }, output_path)
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"\nSaved shard {shard_idx}: {shard_token_count} tokens ({size_gb:.2f} GB)")

print(f"\nDone! {shard_idx + 1} shards in {OUTPUT_DIR}/")
