# -*- coding: utf-8 -*-
# %% [markdown]
# # Lecture N: Sparse Autoencoders
#
# TODO: learning goals
# - Understand why individual neurons may not be the right unit of analysis
# - Train a sparse autoencoder on real model activations
# - Explore the tradeoff between sparsity and reconstruction quality
# - Use pretrained SAEs to find and interpret learned features

# %% [markdown]
# ## 0️⃣ Setup

# %%
from IPython.display import clear_output

try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    import plotly.io as pio
    pio.renderers.default = "colab"
    # !git clone https://github.com/cs221m/cs221m-course.git
    # %cd cs221m-course
    # !uv sync
else:
    import plotly.io as pio
    pio.renderers.default = "plotly_mimetype+png"

clear_output()

# %%
import torch
import numpy as np
from tqdm.auto import tqdm
from nnsight import LanguageModel
from datasets import load_dataset
from IPython.display import display, HTML
import html as html_lib

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# %%
model = LanguageModel("EleutherAI/pythia-70m-v0", device_map=device, dispatch=True)
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# ## 1️⃣ Understanding Model Components by Looking at When They Activate
#
# TODO: prose motivation (rough bullets for now)
# - So far: probing, DAS, interventions — great when you have a specific hypothesis
# - Complementary goal: methods that can surprise us, give general understanding
#   of what components do across many inputs
# - Natural starting point: for each component, ask "when does it activate?"
# - MLP neurons are a natural unit — privileged basis from elementwise nonlinearity
#   (vs attention, which is harder to decompose this way)
# - Let's try it on Pythia-70M

# %% [markdown]
# ### Hooking out neuron activations
#
# For any input text, we can use nnsight to extract MLP neuron activations.
# Let's run a text snippet through the model and look at the raw activation
# values for a specific neuron.

# %%
example_text = "The Chateau de Versailles was built by Louis XIV as a symbol of French royal power."

tokens = tokenizer(example_text, return_tensors="pt", truncation=True, max_length=128)
input_ids = tokens["input_ids"].to(device)
token_strs = [tokenizer.decode([t]) for t in input_ids[0]]

with torch.no_grad():
    with model.trace(input_ids):
        # Post-GELU activations: the input to the down-projection layer
        mlp_activations = model.gpt_neox.layers[3].mlp.dense_4h_to_h.input.save()

# mlp_activations shape: (batch, seq_len, d_mlp)
print(f"MLP activations shape: {mlp_activations.shape}")
print(f"Number of neurons: {mlp_activations.shape[-1]}")

# %%
# Let's look at neuron 609's activation on each token
neuron_609_acts = mlp_activations[0, :, 609].cpu()

print(f"{'Token':<15} {'Activation':>10}")
print("-" * 27)
for tok, act in zip(token_strs, neuron_609_acts):
    print(f"{repr(tok):<15} {act.item():>10.3f}")

# %% [markdown]
# TODO: commentary bullets
# - This model uses a GELU nonlinearity, so "off" neurons have a small negative
#   value rather than exactly zero (as they would with ReLU).
# - The interesting signal is in the positive activations: when a neuron fires
#   strongly positive, it's "responding" to something in the input.
# - Looking at raw numbers is tedious — a common way to visualize this is with
#   a heatmap over the tokens.

# %%
def highlight_text(model, text, layer_idx, neuron_idx, title=None, max_val=4.0):
    """Visualize a single neuron's activation on a text, with tokens colored by activation strength.

    Args:
        max_val: Fixed upper bound for the color scale. Activations >= max_val
                 get full red. Using a fixed scale makes colors comparable across texts.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].to(device)

    with torch.no_grad():
        with model.trace(input_ids):
            hidden = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.input.save()

    acts = hidden[0, :, neuron_idx].cpu()  # (seq_len,)
    token_strs = [tokenizer.decode([t]) for t in input_ids[0]]

    spans = []
    for tok, a in zip(token_strs, acts):
        tok_display = html_lib.escape(tok).replace(" ", "&nbsp;")
        intensity = min(max(0, a.item()) / max_val, 1.0)
        bg = f"rgba(255, 80, 80, {intensity:.2f})"
        spans.append(f'<span style="background-color:{bg};" title="act={a.item():.3f}">{tok_display}</span>')

    header = f"<b>{html_lib.escape(title or '')}</b><br>" if title else ""
    display(HTML(
        f'<div style="font-family: monospace; padding: 8px; line-height: 1.8; border: 1px solid #ddd; border-radius: 4px;">'
        f'{header}{"".join(spans)}'
        f'</div>'
    ))

# %%
highlight_text(
    model,
    "The Chateau de Versailles was built by Louis XIV as a symbol of French royal power.",
    layer_idx=3, neuron_idx=609,
    title="Layer 3, Neuron 609"
)

# %%
highlight_text(
    model,
    "The researchers published their findings in the journal Nature on Tuesday.",
    layer_idx=3, neuron_idx=609,
    title="Layer 3, Neuron 609 (no French content)"
)

# %% [markdown]
# ### ✏️ Exercise 1.1
#
# Try running `highlight_text` on a few more sentences of your choice for both neurons.
# - For neuron 609: try sentences with and without French words. Does it always
#   fire on French content? Does it ever fire on non-French content?
# - For neuron 111: can you figure out what triggers it? Try different kinds of text.

# %%
# YOUR CODE HERE
# highlight_text(model, "your sentence here", layer_idx=3, neuron_idx=609, title="Neuron 609")
# highlight_text(model, "your sentence here", layer_idx=1, neuron_idx=111, title="Neuron 111")

# %% [markdown]
# ### Scaling up: max-activating examples over a corpus
#
# TODO: prose (rough bullets)
# - So far we've been cherry-picking sentences. The French neuron above was
#   identified by Gurnee et al. (2024) — we basically pulled it out of a hat.
# - More generally: run a large volume of pre-training text through the model,
#   record neuron activations on every token, and look at the max-activating examples.
# - This takes a while, so we've pre-computed it. Here's what the collection
#   code looks like (you don't need to run this):
#
# ```python
# dataset = load_dataset("Skylion007/openwebtext", split="train[:100000]")
# texts = [t for t in dataset["text"] if len(t) > 100]
#
# for batch_start in tqdm(range(0, len(texts), batch_size)):
#     batch_texts = texts[batch_start:batch_start + batch_size]
#     tokens = tokenizer(batch_texts, return_tensors="pt", padding=True,
#                        truncation=True, max_length=128)
#     input_ids = tokens["input_ids"].to(device)
#     attention_mask = tokens["attention_mask"].to(device)
#
#     with torch.no_grad():
#         with model.trace(input_ids):
#             hidden = model.gpt_neox.layers[3].mlp.dense_4h_to_h.input.save()
#
#     for batch_idx in range(input_ids.shape[0]):
#         mask = attention_mask[batch_idx].bool()
#         ids = input_ids[batch_idx][mask]
#         acts = hidden[batch_idx][mask].cpu()
#         ...  # save to shards
# ```
#
# The activations are saved in shards of ~1M tokens each. Let's load one
# shard and take a look.

# %%
# TODO: replace with HuggingFace URL once hosted
saved = torch.load("activations/shard_000.pt", weights_only=False)

# all_activations: (total_tokens, d_mlp) — used for both viz and SAE training
# Note: ~1M tokens is small for SAE training but sufficient for a demo
all_activations = saved["activations"].float()
all_input_ids = saved["input_ids"]
seq_lengths = saved["seq_lengths"]

# Reconstruct per-sequence data for visualization
layer3_data = []
offset = 0
for seq_len in seq_lengths:
    ids = all_input_ids[offset:offset + seq_len]
    acts = all_activations[offset:offset + seq_len]
    token_strs = [tokenizer.decode([t]) for t in ids]
    layer3_data.append({"token_strs": token_strs, "input_ids": ids, "activations": acts})
    offset += seq_len

print(f"Loaded {all_activations.shape[0]} tokens across {len(layer3_data)} sequences")
print(f"Activation shape: {all_activations.shape}")

# %% [markdown]
# ### Visualization utilities
#
# We'll use two complementary tools to understand neurons (and later, SAE latents):
# 1. **Max-activating examples**: which tokens in a corpus make it fire most?
# 2. **Top logits**: if this direction fires, what tokens does it promote in the output?
#    (Computed by projecting through the MLP down-projection and unembedding.)

# %%
def get_top_logits(direction, model, layer_idx=3, k=10):
    """Compute effect of a direction in MLP hidden space on output logits.

    Projects direction through W_out (MLP down-projection) and W_unembed.
    Works for both neuron unit vectors and SAE decoder directions.
    """
    with torch.no_grad():
        W_out = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight.data  # (d_model, d_mlp)
        residual_dir = W_out @ direction.to(W_out.device)  # (d_model,)
        W_unembed = model.embed_out.weight.data  # (vocab, d_model)
        logits = W_unembed @ residual_dir  # (vocab,)

    top_vals, top_ids = logits.topk(k)
    bot_vals, bot_ids = logits.topk(k, largest=False)
    top_tokens = [tokenizer.decode([t]) for t in top_ids]
    bot_tokens = [tokenizer.decode([t]) for t in bot_ids]
    return {
        "promoted": list(zip(top_tokens, top_vals.cpu().tolist())),
        "suppressed": list(zip(bot_tokens, bot_vals.cpu().tolist())),
    }

def show_top_logits(direction, model, layer_idx=3, k=10, title=None):
    """Display tokens most promoted/suppressed by a direction in MLP hidden space."""
    result = get_top_logits(direction, model, layer_idx, k)
    if title:
        display(HTML(f"<b>{html_lib.escape(title)}</b>"))
    rows = []
    for i in range(k):
        ptok, pval = result["promoted"][i]
        stok, sval = result["suppressed"][i]
        rows.append(
            f"<tr><td style='color:green'>{html_lib.escape(ptok)}</td><td style='color:green'>{pval:+.2f}</td>"
            f"<td style='padding-left:20px; color:red'>{html_lib.escape(stok)}</td><td style='color:red'>{sval:+.2f}</td></tr>"
        )
    display(HTML(
        f"<table style='font-family:monospace; font-size:0.9em;'>"
        f"<tr><th>Promoted</th><th>Logit</th><th style='padding-left:20px'>Suppressed</th><th>Logit</th></tr>"
        f"{''.join(rows)}</table>"
    ))

def show_top_activating(data, neuron_idx, top_k=8, title=None, context_window=50, max_val=5.0):
    """Display top-activating tokens for a neuron/latent with highlighted context."""
    # Find top activating (sequence_idx, position, activation_value)
    candidates = []
    for seq_idx, seq in enumerate(data):
        acts = seq["activations"][:, neuron_idx]
        for pos in range(len(acts)):
            val = acts[pos].item()
            if val > 0:  # only consider positive activations
                candidates.append((val, seq_idx, pos))

    candidates.sort(key=lambda x: -x[0])

    if title:
        display(HTML(f"<h3 style='text-align:center;'>{html_lib.escape(title)}</h3>"))

    html_parts = []
    seen_seqs = set()
    shown = 0
    for val, seq_idx, pos in candidates:
        if shown >= top_k:
            break
        if seq_idx in seen_seqs:
            continue
        seen_seqs.add(seq_idx)
        shown += 1

        seq = data[seq_idx]
        token_strs = seq["token_strs"]
        acts = seq["activations"][:, neuron_idx]

        start = max(0, pos - context_window)
        end = min(len(token_strs), pos + context_window + 1)

        spans = []
        for i in range(start, end):
            tok = token_strs[i]
            tok_display = html_lib.escape(tok).replace(" ", "&nbsp;")
            if not tok_display:
                tok_display = "&middot;"
            a = max(0, acts[i].item())
            intensity = min(a / max_val, 1.0)

            if i == pos:
                bg = f"rgba(255, 0, 0, {intensity:.2f})"
                spans.append(f'<span style="background-color:{bg}; font-weight:bold; border-bottom: 2px solid red;">{tok_display}</span>')
            else:
                bg = f"rgba(255, 100, 100, {intensity * 0.6:.2f})"
                spans.append(f'<span style="background-color:{bg};">{tok_display}</span>')

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(token_strs) else ""
        html_parts.append(
            f'<div style="font-family: monospace; margin: 6px auto; padding: 6px 8px; '
            f'border-left: 3px solid #ccc; max-width: 900px;">'
            f'<span style="color: #888; font-size: 0.8em;">#{shown} (act={val:.2f})</span><br>'
            f'{prefix}{"".join(spans)}{suffix}'
            f'</div>'
        )

    if shown == 0:
        html_parts.append('<div style="text-align:center; color:#888;">No positive activations found.</div>')

    display(HTML("".join(html_parts)))

# %% [markdown]
# ### ✏️ Exercise 1.2
#
# Look at the max-activating examples for neuron 609 across the corpus.
# Does this confirm your hypothesis from Exercise 1.1?

# %%
show_top_activating(layer3_data, neuron_idx=609, title="Layer 3, Neuron 609 — max activating examples")

# %% [markdown]
# TODO: discuss Geva et al.'s key-value memory interpretation of MLP layers —
# neurons as "keys" that detect patterns, and their output directions as "values"
# that promote specific tokens. We can look at what tokens a neuron's output
# direction promotes by projecting through the unembedding matrix.

# %% [markdown]
# TODO: brief commentary
# - Neuron 609: consistently fires on French/French-origin tokens — monosemantic!
# - Can you write a one-sentence description of what this neuron does?

# %% [markdown]
# ### ✏️ Exercise 1.3
#
# Now explore more neurons on your own. Use the code below to find neurons
# with high max activations, then use `show_top_activating` to inspect them.
#
# For each neuron you look at, try to write a one-sentence description of what
# it responds to. Can you find:
# 1. Another monosemantic neuron (clear single concept)?
# 2. A neuron where the top examples don't share an obvious pattern (polysemantic)?
#
# How many of the neurons you examined were interpretable?

# %%
# Change this index to explore different neurons! (0-2047)
NEURON_IDX = 2

show_top_activating(layer3_data, neuron_idx=NEURON_IDX, title=f"Layer 3, Neuron {NEURON_IDX}")

# %% [markdown]
# ### The problem: polysemanticity and superposition
#
# TODO: transition prose (rough bullets)
# - Some neurons are clean (like our French neuron) — monosemantic
# - But many fire on a grab-bag of unrelated inputs — polysemantic
# - Gurnee et al. (2024) "Neurons in a Haystack": this is widespread,
#   especially in early layers of smaller models
# - Superposition hypothesis (Elhage et al. 2022): model needs more features
#   than neurons, so packs multiple features into overlapping directions
# - Consequence: can't just look at individual neurons to understand the model
# - SAEs aim to undo this — learn an overcomplete dictionary of monosemantic features
#
# <img src="figures/sae_figures/superposition_hypothesis.png" width="700" alt="Superposition hypothesis">
#
# *Figure from [Bricken et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features):
# Under the superposition hypothesis, the neural networks we observe are
# simulations of larger networks where every neuron is a disentangled feature.
# The idealized neurons are projected onto the actual network as "almost orthogonal"
# vectors over the neurons. From the perspective of individual neurons, this
# presents as polysemanticity.*
#
# TODO: prose note
# - SAEs rely on two implicit assumptions:
#   1. **Superposition**: models represent more features than they have neurons,
#      packing them into overlapping directions.
#   2. **Linear representations**: models represent many interesting concepts as
#      linear directions in activation space — so we can train an autoencoder to
#      find them.
# - The least controversial version of (2) is simply the empirical observation
#   that many concepts correspond to directions — we've already seen this in
#   earlier lectures with probing. Stronger variants (e.g., that *all* features
#   are linear, or that these directions are causally meaningful) are more
#   debatable. Carefully defining what "the linear representation hypothesis"
#   actually means and thinking through its implications is an active area of
#   discussion — see [Park et al. (2024)](https://arxiv.org/abs/2311.03658)
#   and [this discussion of the strong feature hypothesis](https://www.alignmentforum.org/posts/tojtPCCRpKLSHBdpn/the-strong-feature-hypothesis-could-be-wrong)
#   for more.

# %% [markdown]
# ## 2️⃣ Sparse Autoencoders
#
# TODO: prose (rough bullets)
# - An SAE learns to reconstruct activations through an *overcomplete* bottleneck
# - More dimensions than the input, but with a sparsity penalty so only a few
#   are active at a time
# - Hope: each dimension corresponds to a single interpretable feature
#
# Training SAEs at scale is expensive, but we can train a small one on the
# MLP neuron activations we already collected. We'll then use Neuronpedia to
# explore features from SAEs trained at much larger scale.
#
# ### Architecture
#
# Given an activation vector $x \in \mathbb{R}^{d}$:
#
# $$z = \text{ReLU}(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}})$$
# $$\hat{x} = W_{\text{dec}} z + b_{\text{dec}}$$
#
# where $W_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}} \times d}$,
# $W_{\text{dec}} \in \mathbb{R}^{d \times d_{\text{sae}}}$,
# and $d_{\text{sae}} \gg d$.
#
# The loss has two terms:
# - **Reconstruction**: $\|x - \hat{x}\|_2^2$ — faithfully represent the input
# - **Sparsity**: $\lambda \|z\|_1$ — most latents should be zero
#
# We also track **L0**: the average number of active latents per input.

# %%
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, lambda_l1=5e-3):
        super().__init__()
        self.lambda_l1 = lambda_l1

        # Encoder: project from model space to (overcomplete) SAE latent space
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: project from SAE latent space back to model space
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize encoder, then set decoder as its transpose (normalized)
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    def encode(self, x):
        return torch.relu(((x - self.b_dec) @ self.W_enc) + self.b_enc)

    def decode(self, z):
        return (z @ self.W_dec) + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_losses(self, x):
        x_hat, z = self.forward(x)
        recon_loss = (x - x_hat).pow(2).mean()
        decoder_norms = self.W_dec.norm(dim=1)
        l1_loss = (z.abs() * decoder_norms).mean()
        loss = recon_loss + self.lambda_l1 * l1_loss
        l0 = (z > 0).float().sum(dim=1).mean().item()
        return {"loss": loss, "recon_loss": recon_loss, "l1_loss": l1_loss, "l0": l0}

# %% [markdown]
# ### Training the SAE on MLP neuron activations
#
# We'll train on the same activations we collected for neuron visualization.
# The SAE will try to decompose each 2048-dim MLP activation vector into a
# sparse combination of learned features.

# %%
D_MODEL = all_activations.shape[1]  # 2048
D_SAE = D_MODEL * 2                 # 4096 (2x overcomplete)
LAMBDA_L1 = 1e-2
LR = 2.5e-3
BATCH_SIZE = 128
NUM_STEPS = 5000

sae = SparseAutoencoder(D_MODEL, D_SAE, lambda_l1=LAMBDA_L1).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
train_data = all_activations.to(device)

# %%
history = []
for step in tqdm(range(NUM_STEPS), desc="Training SAE"):
    indices = torch.randint(0, train_data.shape[0], (BATCH_SIZE,), device=device)
    x = train_data[indices]

    results = sae.compute_losses(x)

    optimizer.zero_grad()
    results["loss"].backward()
    optimizer.step()

    if step % 10 == 0:
        history.append({
            "step": step,
            "recon_loss": results["recon_loss"].item(),
            "l1_loss": results["l1_loss"].item(),
            "l0": results["l0"],
        })

    if step % 200 == 0:
        r = results
        print(f"  Step {step:>4d} | recon={r['recon_loss'].item():.4f} l1={r['l1_loss'].item():.4f} L0={r['l0']:.0f}")

print(f"\nFinal: recon={history[-1]['recon_loss']:.4f} l1={history[-1]['l1_loss']:.4f} L0={history[-1]['l0']:.0f}")

# %% [markdown]
# ### Looking at SAE latent activations
#
# Now let's see what our trained SAE latents respond to. We'll encode our
# corpus through the SAE and look at max-activating examples — just like
# we did for neurons in Section 1.

# %%
# Encode all activations through the SAE in batches
sae.eval()
ENCODE_BATCH = 256
sae_acts_list = []
with torch.no_grad():
    for i in tqdm(range(0, train_data.shape[0], ENCODE_BATCH), desc="Encoding"):
        batch = train_data[i:i + ENCODE_BATCH]
        sae_acts_list.append(sae.encode(batch).cpu())
all_sae_acts = torch.cat(sae_acts_list, dim=0)
del sae_acts_list

print(f"Encoded {all_sae_acts.shape[0]} tokens into {all_sae_acts.shape[1]} SAE latents")

# %% [markdown]
# ### Feature activation frequencies
#
# TODO: prose
# - Not all SAE latents are equally useful — many never activate at all.
#   These are called **dead latents**, and they're a known difficulty with
#   training sparse autoencoders. There are various tricks to minimize this
#   (e.g., resampling dead neurons, auxiliary losses) that we won't cover here.
# - Let's look at how often each feature fires.

# %%
# Fraction of tokens where each feature is active
feature_freq = (all_sae_acts > 0).float().mean(dim=0)  # (d_sae,)

n_dead = (feature_freq == 0).sum().item()
print(f"Dead features (never fire): {n_dead}/{D_SAE} ({n_dead/D_SAE:.0%})")

import plotly.express as px
fig = px.histogram(
    x=feature_freq[feature_freq > 0].log10().numpy(),
    nbins=50,
    labels={"x": "log10(activation frequency)", "y": "Number of features"},
    title="Distribution of SAE feature activation frequencies (excluding dead features)",
)
fig.update_layout(height=350, width=600)
fig.show()

# %%
# Reconstruct per-sequence format for visualization
sae_data = []
offset = 0
for seq in layer3_data:
    seq_len = len(seq["token_strs"])
    sae_data.append({
        "token_strs": seq["token_strs"],
        "input_ids": seq["input_ids"],
        "activations": all_sae_acts[offset:offset + seq_len],
    })
    offset += seq_len

# %%
# Get list of non-dead features
active_latents = (feature_freq > 4e-6).nonzero().squeeze().tolist()
print(f"{len(active_latents)} active features (out of {D_SAE})")

# %%
# Pick a random active latent — re-run this cell to explore!
import random
LATENT_IDX = random.choice(active_latents)

show_top_activating(sae_data, neuron_idx=LATENT_IDX, title=f"SAE Latent {LATENT_IDX} (freq={feature_freq[LATENT_IDX]:.2e})")

# %% [markdown]
# ### ✏️ Bonus Exercise: Hyperparameter exploration
#
# The quality of learned SAE features depends heavily on — in addition to
# typical ML hyperparameters — the **sparsity coefficient** ($\lambda$) and
# **dictionary size** ($d_{\text{sae}}$).
#
# Try changing these and re-running the training + exploration cells above:
# - What happens to reconstruction quality and L0 as you increase/decrease $\lambda$?
# - What about increasing $d_{\text{sae}}$ (e.g., 4x or 8x overcomplete)?
# - We are also interested in whether the learned features are more interpretable,
#   though in this small-scale setting where we are significantly under-training
#   the SAE, it may be hard to get a clear signal.
#
# Evaluating feature interpretability rigorously is itself an open question
# that we won't dive into here — but your qualitative impressions are valuable.

# %%
# YOUR CODE HERE

# %% [markdown]
# ## 3️⃣ Exploring SAE Features at Scale
#
# TODO: prose (rough bullets)
# - We trained a small SAE — but are SAE features actually more interpretable
#   than the MLP neurons we started with?
# - Evaluating the interpretability of model components is pretty hard. The most
#   common approach involves having two language models play a game:
#   1. The first looks at activating examples and comes up with a text description
#      of when the component activates.
#   2. The second takes the description and tries to predict activating examples
#      on held-out text.
#   A component is considered more interpretable if the second model's predictions
#   are more accurate. This was first introduced by
#   [Bills et al. (2023), "Language models can explain neurons in language models"](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html).
# - When Anthropic trained large-scale SAEs on MLP neurons
#   ([Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features)),
#   they found that SAE features scored significantly higher on this
#   auto-interpretability metric than individual MLP neurons.
#
# <img src="figures/sae_figures/scaling_monosemanticity_mlp_vs_sae.png" width="600">
#
# *Figure from [Templeton et al. (2024), "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/).
# When Anthropic scaled SAEs up to MLP neurons from Claude 3 Sonnet, they
# performed this sort of automated comparison and found that SAE features did
# in fact appear to be more interpretable than the underlying MLP neurons.*
#
# - This gives us confidence that SAEs are finding something real. Let's use
#   Neuronpedia to explore features from SAEs trained at much larger scale
#   than our demo.
#
# ### Exploring features on Neuronpedia
#
# TODO: exercises using Neuronpedia/SAELens to find interesting features,
# steering with SAE features, etc.
