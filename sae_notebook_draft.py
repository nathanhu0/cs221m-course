# -*- coding: utf-8 -*-
# %% [markdown]
# # Sparse Autoencoders
#
# ### ✍ Learning goals
#
# * Understand how we can try to interpret model components (MLP neurons,
#   learned SAE features) by looking at their activating examples and
#   logit effects.
# * Build intuition for how model internals can have structure but still be
#   messy — and understand how the **superposition hypothesis** is one possible
#   explanation for the apparent polysemanticity of MLP neurons.
# * Get hands-on experience training a **sparse autoencoder** as one tool for
#   finding more monosemantic features in a model.
# * [TODO] Gain familiarity with resources like **SAELens** and **Neuronpedia** and
#   how we can use them to explore model internals.

# %% [markdown]
# ## 0️⃣ Setup
#
# In this notebook we'll study [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m-v0),
# a small (70M param) language model trained by [EleutherAI](https://www.eleuther.ai/).

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
    !git clone https://github.com/cs221m/cs221m-course.git
    %cd cs221m-course
    !uv sync
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
model = LanguageModel("EleutherAI/pythia-70m-v0", device_map=device, dispatch=True, torch_dtype=torch.bfloat16)
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# ## 1️⃣ Studying MLP neurons via activating examples
#
# Methods like probing, DAS, and interventions are powerful for deeply studying
# specific behaviors — but they require a specific hypothesis and
# behavior-specific datasets up front. We might also want more open-ended ways
# to study model internals — methods that can *surprise us*, without needing to
# know what to look for ahead of time.
#
# Although this sounds hard, remember that we can always just look at what any
# part of the model is doing on any input. We've previously visualized attention patterns, but today we will start with MLP neurons. 

# %% [markdown]
# ### Hooking out neuron activations
#
# For any input text, we can use nnsight to extract MLP neuron activations.
# Let's run a text snippet through the model and look at the raw activation
# values for a specific neuron.

# %%
example_text = "The Chateau de Versailles was built by Louis XIV as a symbol of French royal power."

tokens = tokenizer(example_text, return_tensors="pt")
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
# This model uses a GELU nonlinearity, so "off" neurons have a small negative
# value rather than exactly zero (as they would with ReLU). The interesting
# signal is in the positive activations — when a neuron fires strongly, it's
# responding to something in the input. 
#
# Looking at raw numbers is tedious. A common way to visualize activation
# patterns is as a heatmap over tokens. Below we define a few helpers:
#
# - `token_heatmap(tokens, activations)` — displays tokens colored by
#   activation strength.
# - `get_mlp_neuron_activations(model, text, layer_idx, neuron_idx)` — runs
#   text through the model and returns the activations for a single MLP neuron.
# - `color_tokens_by_mlp_neuron(...)` — combines the two: runs the model and
#   displays the heatmap.

# %%
def merge_tokens(token_ids, activations, tokenizer):
    """Merge multi-token characters (e.g. emojis) into single display tokens.

    Returns (merged_strs, merged_acts, raw_to_merged) where:
    - merged_strs: list of decoded strings (one per merged token)
    - merged_acts: list of averaged activation values
    - raw_to_merged: list mapping each raw token index to its merged index
    """
    merged_strs, merged_acts = [], []
    raw_to_merged = [0] * len(token_ids)
    i = 0
    while i < len(token_ids):
        for j in range(i + 1, len(token_ids) + 1):
            decoded = tokenizer.decode(token_ids[i:j])
            if "\ufffd" not in decoded:
                merged_idx = len(merged_strs)
                avg_act = activations[i:j].float().mean().item()
                merged_strs.append(decoded)
                merged_acts.append(avg_act)
                for k in range(i, j):
                    raw_to_merged[k] = merged_idx
                i = j
                break
        else:
            raw_to_merged[i] = len(merged_strs)
            merged_strs.append(tokenizer.decode([token_ids[i]]))
            merged_acts.append(float(activations[i]))
            i += 1
    return merged_strs, merged_acts, raw_to_merged


def token_heatmap(token_strs, activations, max_val=4.0, title=None, highlight_idx=None, return_html=False):
    """Display tokens colored by activation strength.

    Args:
        token_strs: list of token strings
        activations: list/tensor of activation values (one per token)
        max_val: activations >= max_val get full color intensity
        title: optional title displayed above the heatmap
        highlight_idx: if set, bold + underline the token at this index
        return_html: if True, return HTML string instead of displaying
    """
    spans = []
    for i, (tok, a) in enumerate(zip(token_strs, activations)):
        tok_display = html_lib.escape(tok).replace("\n", "↵") or "&middot;"
        intensity = min(max(0, float(a)) / max_val, 1.0)
        if i == highlight_idx:
            bg = f"rgba(255, 0, 0, {intensity:.2f})"
            spans.append(f'<span style="background-color:{bg}; font-weight:bold; border-bottom: 2px solid red;" title="act={float(a):.3f}">{tok_display}</span>')
        else:
            bg = f"rgba(255, 80, 80, {intensity:.2f})"
            spans.append(f'<span style="background-color:{bg};" title="act={float(a):.3f}">{tok_display}</span>')

    header = f"<b>{html_lib.escape(title or '')}</b><br>" if title else ""
    html_str = (
        f'<div style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word; '
        f'padding: 8px; line-height: 1.8; border: 1px solid #ddd; border-radius: 4px; max-width: 900px;">'
        f'{header}{"".join(spans)}'
        f'</div>'
    )
    if return_html:
        return html_str
    display(HTML(html_str))


def get_mlp_neuron_activations(model, text, layer_idx, neuron_idx):
    """Run text through the model and return (token_strs, activations) for one MLP neuron."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].to(device)

    with torch.no_grad():
        with model.trace(input_ids):
            hidden = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.input.save()

    acts = hidden[0, :, neuron_idx].cpu()
    token_strs, acts, _ = merge_tokens(input_ids[0].cpu(), acts, tokenizer)
    return token_strs, acts


def color_tokens_by_mlp_neuron(model, text, layer_idx, neuron_idx, title=None, max_val=4.0):
    """Run text through the model and display a heatmap of one MLP neuron's activations."""
    token_strs, acts = get_mlp_neuron_activations(model, text, layer_idx, neuron_idx)
    token_heatmap(token_strs, acts, max_val=max_val, title=title)

# %% [markdown]
#  Let's try visualizing the earlier sentence and a few others:
# %%
color_tokens_by_mlp_neuron(
    model,
    "The Chateau de Versailles was built by Louis XIV as a symbol of French royal power.",
    layer_idx=3, neuron_idx=609,
    title="Layer 3, Neuron 609"
)


color_tokens_by_mlp_neuron(
    model,
    "Here is sentence that doesn't contain any words in French.",
    layer_idx=3, neuron_idx=609,
    title="Layer 3, Neuron 609 (no French content)"
)


# %% [markdown]
# ### ✏️ Exercise 1.1
#
# Try running `color_tokens_by_mlp_neuron` on a few more sentences of your choice for neuron L3 N609. Try sentences with and without French words. Does it always fire on French content? Does it ever fire on non-French content? What do you think this neuron is doing?


# %%
# YOUR CODE HERE
# color_tokens_by_mlp_neuron(model, "your sentence here", layer_idx=3, neuron_idx=609, title="Neuron 609")


# %% [markdown]
# ### Scaling up: max-activating examples over a dataset
#
# So far we've been handwriting sentences and pulled the specific neuron we
# want to study out of a hat. The French neuron above was identified by
# Gurnee et al. (2024) by trained sparse probes to identify
# interesting neurons, which we will not cover here. Instead we will explore
# another common workflow to screen for interesting neurons. We run a large
# volume of text through the model, recording neuron activations on every
# token. Then for each neuron, we can look at the max-activating examples.
# Let's collect ~1M tokens of MLP activations from OpenWebText. (We'll reuse
# these for SAE training later too.)

# %%
LAYER_IDX = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
N_TEXTS = 10000  # ~1M tokens (128 tokens/text)

dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
texts = []
for example in dataset:
    if len(example["text"]) > 100:
        texts.append(example["text"])
    if len(texts) >= N_TEXTS:
        break
print(f"Loaded {len(texts)} texts")

# %%
# Collect MLP neuron activations per sequence
layer3_data = []

for batch_start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Collecting activations"):
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
        acts = hidden[batch_idx][mask].cpu()
        layer3_data.append({"input_ids": ids, "activations": acts})

total_tokens = sum(len(seq["input_ids"]) for seq in layer3_data)
print(f"Collected {total_tokens} tokens across {len(layer3_data)} sequences")

# %% [markdown]
# Now that we have this dataset of neuron activations, we can write some code
# to find the max-activating inputs for each neuron and visualize them using
# our heatmap helper from before. This logic is all wrapped together in
# `find_and_show_top_neuron_examples` below.

# %%
def show_max_activating_examples(examples, title=None, context_window=50, max_val=5.0):
    """Display a ranked list of max-activating examples as token heatmaps.

    Args:
        examples: list of (activation_value, token_strs, acts_1d, pos) tuples
        title: optional title
        context_window: number of tokens to show around the max-activating token
        max_val: activations >= max_val get full color intensity
    """
    if title:
        display(HTML(f"<h3 style='text-align:center;'>{html_lib.escape(title)}</h3>"))

    if not examples:
        display(HTML('<div style="text-align:center; color:#888;">No positive activations found.</div>'))
        return

    html_parts = []
    for rank, (val, token_strs, acts, pos) in enumerate(examples):
        # Truncate to max_chars centered on the max-activating token
        max_chars = 260  # ~3 lines at 900px monospace
        # Expand outward from pos until we hit the char budget
        start, end = pos, pos + 1
        total_chars = len(token_strs[pos]) if pos < len(token_strs) else 0
        while True:
            expanded = False
            if start > 0 and total_chars + len(token_strs[start - 1]) <= max_chars:
                start -= 1
                total_chars += len(token_strs[start])
                expanded = True
            if end < len(token_strs) and total_chars + len(token_strs[end]) <= max_chars:
                total_chars += len(token_strs[end])
                end += 1
                expanded = True
            if not expanded:
                break

        window_tokens = token_strs[start:end]
        window_acts = acts[start:end]
        highlight_idx = pos - start

        heatmap_html = token_heatmap(window_tokens, window_acts, max_val=max_val, highlight_idx=highlight_idx, return_html=True)

        html_parts.append(
            f'<div style="margin: 6px auto; max-width: 900px;">'
            f'<span style="color: #888; font-size: 0.8em;">#{rank+1} (act={val:.2f})</span> '
            f'{heatmap_html}'
            f'</div>'
        )

    display(HTML("".join(html_parts)))


def show_top_logits(direction, model, tokenizer, k=5):
    """Show the top and bottom k tokens promoted by a direction, as two horizontal lines."""
    with torch.no_grad():
        W_unembed = model.embed_out.weight.data
        logits = W_unembed @ direction.to(W_unembed.device)

    top_vals, top_ids = logits.topk(k)
    bot_vals, bot_ids = logits.topk(k, largest=False)

    def _render_tokens(ids, vals, bg_color):
        spans = []
        for t, v in zip(ids, vals):
            tok = html_lib.escape(tokenizer.decode([t]).strip() or "·")
            spans.append(f'<span style="background:{bg_color}; padding:1px 5px; margin:0 2px; border-radius:3px;">{tok} <span style="color:#888;">{v:+.2f}</span></span>')
        return " ".join(spans)

    promoted = _render_tokens(top_ids, top_vals, "rgba(100,200,100,0.15)")
    suppressed = _render_tokens(bot_ids, bot_vals, "rgba(255,100,100,0.15)")
    display(HTML(
        f'<div style="font-family: monospace; font-size: 0.85em; line-height: 2; max-width: 900px;">'
        f'<span style="color:#999;">Promotes:</span> {promoted}<br>'
        f'<span style="color:#999;">Suppresses:</span> {suppressed}'
        f'</div>'
    ))


def find_and_show_top_neuron_examples(data, neuron_idx, top_k=8, layer_idx=LAYER_IDX, show_logits=True, **kwargs):
    """Search collected activation data for top-k examples of a neuron and display them."""
    seq_maxes = []
    for seq_idx, seq in enumerate(data):
        acts = seq["activations"][:, neuron_idx]
        val = acts.max().item()
        if val > 0:
            pos = acts.argmax().item()
            seq_maxes.append((val, seq_idx, pos))
    seq_maxes.sort(key=lambda x: -x[0])

    examples = []
    for val, si, pos in seq_maxes[:top_k]:
        ids = data[si]["input_ids"]
        acts = data[si]["activations"][:, neuron_idx]
        token_strs, merged_acts, raw_to_merged = merge_tokens(ids, acts, tokenizer)
        examples.append((val, token_strs, merged_acts, raw_to_merged[pos]))
    # Render title first, then logits, then examples
    title = kwargs.pop("title", None)
    if title:
        display(HTML(f"<h3 style='text-align:center;'>{html_lib.escape(title)}</h3>"))

    if show_logits:
        with torch.no_grad():
            W_out = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight.data
            direction = W_out[:, neuron_idx]
        show_top_logits(direction, model, tokenizer)

    show_max_activating_examples(examples, **kwargs)

# %% [markdown]
# ### ✏️ Exercise 1.2
#
# Look at the max-activating examples for neuron L3.N609 across the corpus.
# Does this confirm your hypothesis from Exercise 1.1?

# %%
find_and_show_top_neuron_examples(layer3_data, 609, title="Layer 3, Neuron 609 — max activating examples", show_logits=False)

# %% [markdown]
# In addition to seeing *when* a neuron fires, we can ask what it *does* to the
# model's output. We can apply the same logit lens idea from earlier lectures
# to the specific direction a neuron writes — projecting its output direction
# through the unembedding matrix to see which tokens it promotes or suppresses.
#
# This view of MLP neurons as keys (responding to specific contexts) and values
# (writing specific information to the residual stream, which we can
# approximately decode with logit lens) was first proposed by
# [Geva et al. (2021), "Transformer Feed-Forward Layers Are Key-Value Memories"](https://arxiv.org/abs/2012.14913).
#

# %%
# What tokens does neuron 609 promote/suppress in the output?
with torch.no_grad():
    W_out = model.gpt_neox.layers[3].mlp.dense_4h_to_h.weight.data  # (d_model, d_mlp)
    neuron_output_dir = W_out[:, 609]  # (d_model,) — this neuron's "value" direction
    W_unembed = model.embed_out.weight.data  # (vocab, d_model)
    logit_effects = W_unembed @ neuron_output_dir  # (vocab,)

top_k = 10
top_vals, top_ids = logit_effects.topk(top_k)
bot_vals, bot_ids = logit_effects.topk(top_k, largest=False)

print("Neuron 609 — tokens PROMOTED when it fires:")
for tok_id, val in zip(top_ids, top_vals):
    print(f"  {val.item():+.3f}  {repr(tokenizer.decode([tok_id]))}")

print("\nNeuron 609 — tokens SUPPRESSED when it fires:")
for tok_id, val in zip(bot_ids, bot_vals):
    print(f"  {val.item():+.3f}  {repr(tokenizer.decode([tok_id]))}")

# %% [markdown]
# ### From observation to evidence
#
# This qualitative approach — observing how model components like neurons behave
# across different inputs — is a great *starting point* for *building hypotheses*
# about what the model is doing. 
# For this specific neuron, we've seen that its max-activating examples are
# French text, and that its output direction promotes French tokens. This is
# pretty good starting evidence that this neuron detects French text and aids
# the model's computation when modeling French language. But this is all
# observational, and we should do more work to confirm. 

# ### ✏️ Exercise 1.2b
#
# What experiments would you run to confirm this? Write your answer below
# before reading on. (Hint: think interventions 😉)

# %% [markdown]
# > FILL IN YOUR ANSWER HERE

# %% [markdown]
# ### Additional "French Neuron" validation 
#
# This "french neuron" was found by [Gurnee et al. (2024), "Finding Neurons in a Haystack"](https://arxiv.org/abs/2305.01610). They further validate its role with 2 sets of experiments.
#
# <img src="figures/sae_figures/french_neuron_results.png" width="900" alt="Gurnee et al. French Neuron results">

#
# 1. **Reliable detection** (left and middle panels): They evaluated this neuron's
#    activations on a large multilingual dataset and found that the
#    distributions of activations on French vs. non-French tokens are
#    well-separated (left). The middle panel shows that this separation holds even
#    for individual tokens that appear in both French and non-French contexts.
#
# 2. **Causal effect** (right panel): They **ablated** the neuron (set its
#    activation to zero) and measured the change in loss, bucketed by language.
#    Ablating the neuron  causes a large increase in loss specifically on French
#    text, with minimal effect on other languages. This is evidence that
#    the neuron plays a causal role in the model's processing of French.

# %% [markdown]
# ### ✏️ Exercise 1.3
#
# Use the code below to explore max-activating examples and logits for
# different neurons. How often do you find neurons with clean, interpretable
# patterns vs. neurons that seem to fire on a grab-bag of unrelated inputs?
#
# (The logits are often uninformative, but occasionally very revealing — don't
# worry if they don't make sense for most neurons.)

# %%
# Change this index to explore different neurons! (0-2047)
NEURON_IDX = 1

find_and_show_top_neuron_examples(layer3_data, NEURON_IDX, title=f"Layer 3, Neuron {NEURON_IDX}")

# %% [markdown]
# ### 🧠 Takeaways
#
# We've seen that we can start to get a feel for what model components do just
# by looking at their activating examples. But while some neurons have clean
# structure — like the French neuron — many are less interpretable, and appear
# to fire on a grab-bag of unrelated inputs.
# [Gurnee et al. (2024)](https://arxiv.org/abs/2305.01610) show this is
# widespread. Neurons with clean, single-concept responses are often called
# **monosemantic**, while the grab-bag ones are called **polysemantic**.

# %% [markdown]
# ## 2️⃣ Using Sparse Autoencoders to Find Interpretable Features
#
# ### Motivation: the superposition hypothesis and the linear representation hypothesis
#
# #### Superposition as an explanation for polysemanticity
#
# In the previous section we saw that while some MLP neurons seemed very
# monosemantic, many other neurons appeared **polysemantic**, activating on
# many different inputs. One hypothesis for why this is the case is the
# **superposition hypothesis**. To effectively model language, we expect that
# a model would want to represent many more concepts (millions? billions? even
# more?) than it has dimensions or MLP neurons (thousands). If these concepts
# are sparsely activating, the model can pack them into overlapping directions
# — representing more features than it has dimensions. Under this view, the
# network we observe is a noisy, low-dimensional projection of a much larger,
# sparser network where each neuron represents a single clean feature.
#
# <img src="figures/sae_figures/superposition_hypothesis.png" width="700" alt="Superposition hypothesis">
#
# *Figure from [Bricken et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features)*
#
# #### The linear representation hypothesis
#
# Sparse Autoencoders are additionally motivated by the **linear representation hypothesis**
# — the idea that neural networks represent meaningful concepts as directions
# in activation space. We've already seen some evidence for this when we
# trained linear probes in a prior lecture. We know this is true to some
# degree, but nailing down what exactly "linear representation" means and whether the hypothesis is a statement about some vs. all
# features, is an active area of discussion. See
# [Park et al. (2024)](https://arxiv.org/abs/2311.03658) and
# [Smith (2024)](https://www.alignmentforum.org/posts/tojtPCCRpKLSHBdpn/the-strong-feature-hypothesis-could-be-wrong).

# ### The Sparse Autoencoder

#
# These two hypotheses together give us a strong prior about the structure
# we'd expect to find in a model's activations. When you have a hypothesis
# about structure in your data and want to find it in an unsupervised way, a
# classic ML tool is the **autoencoder**. An autoencoder learns to map each
# input back to itself, but is constrained to pass through some intermediate
# representation — a "bottleneck" — that you design to have the structure
# you're looking for.
#
# So what structure do our hypotheses suggest? The superposition hypothesis
# says the model represents many more concepts than it has dimensions — so
# the intermediate representation should be *higher-dimensional* than the
# input, but *sparse* (only a few concepts active at a time). The linear
# representation hypothesis says concepts are directions — so the mapping to
# and from this intermediate representation can be *linear*, giving us a
# single-layer encoder and decoder.
#
# This is exactly what a **sparse autoencoder (SAE)** is. More broadly, this
# falls under the umbrella of **sparse dictionary learning** — decomposing
# data into sparse linear combinations of elements from a learned
# overcomplete dictionary. Sparse dictionary learning has been a common idea
# in neuroscience
# ([Olshausen & Field, 1997](https://doi.org/10.1016/S0042-6989(97)00169-7)).
# The earliest work I could find applying it to language models is
# [Arora et al. (2018)](https://arxiv.org/abs/1601.03764), who used sparse
# coding to recover polysemous word senses from word embeddings. Some of the
# first work proposing SAEs as a solution to polysemanticity in neural
# network internals is
# [Cunningham et al. (2023)](https://arxiv.org/abs/2309.08600) and
# [Bricken et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features),
# and this has since been scaled up to large production models.
#
# The hope is that each learned dimension in the SAE's intermediate
# representation corresponds to a single interpretable feature. Training
# SAEs at scale is expensive, but we can train a small one on the MLP
# activations we already collected.
#
# ### Architecture
#
# Given an activation vector $x \in \mathbb{R}^{d}$, the encoder maps it
# to a sparse, higher-dimensional latent representation $z$:
#
# $$z = \text{ReLU}(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}})$$
#
# The decoder reconstructs the original activation from this sparse code:
#
# $$\hat{x} = W_{\text{dec}} z + b_{\text{dec}}$$
#
# where $W_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}} \times d}$,
# $W_{\text{dec}} \in \mathbb{R}^{d \times d_{\text{sae}}}$,
# and $d_{\text{sae}} \gg d$.
#
# The loss has two terms:
# - **Reconstruction**: $\|x - \hat{x}\|_2^2$ — faithfully represent the input
# - **Sparsity**: $\lambda \|z\|_1$ — incentivize sparse latents
#
# Although we ultimately care about **L0** (the number of non-zero latents),
# L0 is not differentiable. SAEs commonly penalize the **L1 norm** instead,
# as an easier-to-optimize objective that still incentivizes sparsity.
#
# Note: what we describe here is a simple "vanilla" SAE. If you are
# curious about the cutting edge of SAE training details,
# [Gemma Scope 2 (2025)](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf)
# and [Anthropic (2025), Appendix D](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#appendix-ml-details)
# are good starting points.

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
        # Normalized MSE: MSE / Var(x), so loss is scale-invariant
        recon_loss = (x - x_hat).pow(2).mean() / (x - x.mean(dim=0)).pow(2).mean()
        decoder_norms = self.W_dec.norm(dim=1)
        l1_loss = (z * decoder_norms).sum(dim=1).mean()  # sum over features, mean over batch
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
# Build flat tensor for SAE training from per-sequence data
train_data = torch.cat([seq["activations"] for seq in layer3_data], dim=0)
print(f"Training data: {train_data.shape} (on CPU, batches moved to {device} during training)")

D_MODEL = train_data.shape[1]  # 2048
D_SAE = D_MODEL * 4            # 8192 (4x overcomplete)
LAMBDA_L1 = 3e-3
LR = 5e-3
BATCH_SIZE = 128
NUM_STEPS = 10000

sae = SparseAutoencoder(D_MODEL, D_SAE, lambda_l1=LAMBDA_L1).to(device=device, dtype=torch.bfloat16)

# # Initialize b_dec to the mean of the first batch
# with torch.no_grad():
#     sae.b_dec.data = train_data[:BATCH_SIZE].to(device=device, dtype=torch.bfloat16).mean(dim=0)

optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

WARMUP_STEPS = int(0.05 * NUM_STEPS)
def lr_schedule(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS  # linear warmup
    # cosine decay to 0
    progress = (step - WARMUP_STEPS) / (NUM_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

# %%
perm = torch.randperm(train_data.shape[0])
history = []
for step in tqdm(range(NUM_STEPS), desc="Training SAE"):
    # Cycle through shuffled data
    start = (step * BATCH_SIZE) % train_data.shape[0]
    x = train_data[perm[start:start + BATCH_SIZE]].to(device)

    results = sae.compute_losses(x)

    optimizer.zero_grad()
    results["loss"].backward()
    optimizer.step()
    scheduler.step()

    if step % 10 == 0:
        history.append({
            "step": step,
            "loss": results["loss"].item(),
            "recon_loss": results["recon_loss"].item(),
            "l1_loss": results["l1_loss"].item(),
            "l0": results["l0"],
        })

    if step % 200 == 0:
        r = results
        print(f"  Step {step:>4d} | loss={r['loss'].item():.4f} recon={r['recon_loss'].item():.4f} l1={r['l1_loss'].item():.4f} L0={r['l0']:.0f}")

print(f"\nFinal: loss={history[-1]['loss']:.4f} recon={history[-1]['recon_loss']:.4f} l1={history[-1]['l1_loss']:.4f} L0={history[-1]['l0']:.0f}")

# %% [markdown]
# ### Looking at SAE latent activations
#
# Now let's see what our trained SAE latents respond to. We'll encode our
# corpus through the SAE and look at max-activating examples — just like
# we did for neurons in Section 1.

# %%
# Stream through data: encode each sequence, track feature frequencies and
# top-k activating examples per latent. We do NOT store all SAE activations
# (that would be D_SAE * n_tokens floats — too much memory).
import heapq

sae.eval()
TOP_K_SAE = 10
MAX_SEQS = len(layer3_data)  # reduce for faster dev, e.g. 500

# Running stats
feature_fire_count = torch.zeros(D_SAE, device=device)
total_tokens_seen = 0
sae_top_examples = {i: [] for i in range(D_SAE)}  # min-heaps

with torch.no_grad():
    for seq_idx, seq in enumerate(tqdm(layer3_data[:MAX_SEQS], desc="Encoding through SAE")):
        x = seq["activations"].to(device)
        z = sae.encode(x)  # (seq_len, d_sae) — stays on GPU
        seq_len = z.shape[0]

        # Update frequency counts (on GPU)
        feature_fire_count += (z > 0).float().sum(dim=0)
        total_tokens_seen += seq_len

        # Find max activation per latent (on GPU)
        max_vals, max_pos = z.max(dim=0)  # (d_sae,), (d_sae,)
        max_vals_cpu = max_vals.cpu()
        max_pos_cpu = max_pos.cpu()
        text_key = tuple(seq["input_ids"][:20].tolist())

        # Figure out which latents need storing (CPU-side scalar checks)
        need_transfer = []
        for latent_idx in range(D_SAE):
            val = max_vals_cpu[latent_idx].item()
            if val <= 0:
                continue
            heap = sae_top_examples[latent_idx]
            if len(heap) < TOP_K_SAE or val > heap[0][0]:
                existing_texts = {tuple(e[2][0][:20].tolist()) for e in heap}
                if text_key not in existing_texts:
                    need_transfer.append(latent_idx)

        # Single batched GPU→CPU transfer for all needed columns
        if need_transfer:
            cols = torch.tensor(need_transfer, device=device)
            z_subset = z[:, cols].cpu()  # (seq_len, len(need_transfer))

            for i, latent_idx in enumerate(need_transfer):
                val = max_vals_cpu[latent_idx].item()
                pos = max_pos_cpu[latent_idx].item()
                entry = (seq["input_ids"], z_subset[:, i], pos)
                heap = sae_top_examples[latent_idx]
                if len(heap) < TOP_K_SAE:
                    heapq.heappush(heap, (val, seq_idx, entry))
                else:
                    heapq.heapreplace(heap, (val, seq_idx, entry))

# Convert heaps to sorted lists, merging tokens for display
for latent_idx in tqdm(range(D_SAE), desc="formatted activating examples"):
    merged = []
    for val, _seq_idx, (ids, acts, pos) in sorted(sae_top_examples[latent_idx], key=lambda x: -x[0]):
        token_strs, merged_acts, raw_to_merged = merge_tokens(ids, acts, tokenizer)
        merged.append((val, token_strs, merged_acts, raw_to_merged[pos]))
    sae_top_examples[latent_idx] = merged

feature_freq = (feature_fire_count / total_tokens_seen).cpu()
print(f"Encoded {total_tokens_seen} tokens, {D_SAE} latents")

# %% [markdown]
#
# Let's take a look at the distribution of feature frequencies. Note that
# they are all pretty sparse. One problem when training SAEs is **dead
# features** — features which are never active. More advanced training
# recipes use a couple different tricks to try to minimize the number of
# dead features.

# %%
n_dead = (feature_freq == 0).sum().item()
print(f"Dead features (never fire): {n_dead}/{D_SAE} ({n_dead/D_SAE:.0%})")

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 3))
plt.hist(feature_freq[feature_freq > 0].log10().numpy(), bins=50)
plt.xlabel("log10(activation frequency)")
plt.ylabel("Number of features")
plt.title("SAE feature activation frequencies (excluding dead features)")
plt.tight_layout()
plt.show()


# %% [markdown]
#
# Now lets look at activating text dashboards for features in the SAE we've trained. How interpretable do they seem compared to MLP neurons?

# %%
# Get list of non-dead features
active_latents = (feature_freq > 0).nonzero().squeeze().tolist()
print(f"{len(active_latents)} active features (out of {D_SAE})")

def show_sae_latent_dashboard(sae, sae_top_examples, feature_freq, latent_idx, layer_idx=LAYER_IDX):
    """Display a dashboard for an SAE latent: title, logits, and max-activating examples."""
    title = f"SAE Latent {latent_idx} (freq={feature_freq[latent_idx]:.2e})"
    display(HTML(f"<h3 style='text-align:center;'>{html_lib.escape(title)}</h3>"))

    examples = sae_top_examples.get(latent_idx, [])
    max_val = max((val for val, *_ in examples), default=5.0)
    show_max_activating_examples(examples, max_val=max_val)

# %%
# Pick a random active latent — re-run this cell to explore!
import random
LATENT_IDX = random.choice(active_latents)

show_sae_latent_dashboard(sae, sae_top_examples, feature_freq, LATENT_IDX)

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


# %%
# YOUR CODE HERE

# %% [markdown]
# ### Are SAE features more interpretable than MLP neurons?
#
# We trained a small SAE — but are SAE features actually more interpretable
# than the MLP neurons we started with?
#
# Evaluating interpretability can be tricky. A common approach uses two
# language models to play a game: the first looks at activating examples and
# writes a description of when the component activates; the second takes
# that description and tries to predict activations on held-out text. A
# component is more interpretable if the second model's predictions are more
# accurate. This "autointerp" metric was introduced by
# [Bills et al. (2023)](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html).
#
# <img src="figures/sae_figures/scaling_monosemanticity_mlp_vs_sae.png" width="600">
#
# *Figure from [Templeton et al. (2024), "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/).*
#
# When Anthropic scaled SAEs to Claude 3 Sonnet
# ([Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)),
# they found that SAE features scored significantly higher on this
# auto-interpretability metric than individual MLP neurons.
#

# %%
# ### Exploring features on Neuronpedia
#
# TODO: exercises using Neuronpedia/SAELens to find interesting features,
# steering with SAE features, etc.

# %%
