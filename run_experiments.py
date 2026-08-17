"""
Experiments for "Every Symbol Explained" -- Sec. XV (Measuring Next-Token Uncertainty).

Produces every [RUN] value and both figures for the paper.

Setup (once, ~5 min):
    pip install torch transformers matplotlib
Run (CPU is fine, ~5-10 min total):
    python run_experiments.py
Outputs:
    results.csv          -> numbers for Table (entropy + nucleus size)  [Exp 1 & 3]
    temp_sweep.pdf       -> Fig. temperature sweep                      [Exp 2]
    repetition.pdf       -> Fig. repetition vs temperature              [Exp 4]
    console output       -> sentences to paste into Sec. XV prose
"""

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(0)

print("Loading GPT-2 small...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

CONTEXTS = [
    "The capital of France is",
    "2 + 2 =",
    "I went to the",
    "The",
    "Once upon a time",
]


def next_token_logits(text: str) -> torch.Tensor:
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
    return out.logits[0, -1, :]  # [vocab]


def entropy_bits(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(-(p * torch.log2(p)).sum())


def nucleus_size(probs: torch.Tensor, p: float = 0.9) -> int:
    sorted_p, _ = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=0)
    return int((cum < p).sum().item()) + 1  # smallest set with cum >= p


# ---------------- Experiment 1 & 3: entropy and nucleus size per context ----
print("\n=== Experiment 1 & 3: entropy and nucleus size (Table) ===")
rows = []
for ctx in CONTEXTS:
    logits = next_token_logits(ctx)
    probs = F.softmax(logits, dim=-1)
    H = entropy_bits(probs)
    n = nucleus_size(probs, 0.9)
    top = tok.decode(int(torch.argmax(probs)))
    rows.append((ctx, H, n, top))
    print(f"  {ctx!r:35s}  H = {H:6.2f} bits   |nucleus(0.9)| = {n:6d}   top: {top!r}")

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["context", "entropy_bits", "nucleus_size_p0.9", "top_token"])
    w.writerows(rows)
print("  -> results.csv written. Paste H and |nucleus| into Table~\\ref{tab:entropy}.")

# ---------------- Experiment 2: temperature sweep ---------------------------
print("\n=== Experiment 2: temperature sweep (Fig. temp_sweep.pdf) ===")
SWEEP_CTX = "I went to the"  # medium-entropy context; change if you prefer
logits = next_token_logits(SWEEP_CTX)
temps = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
Hs = []
for T in temps:
    probs = F.softmax(logits / T, dim=-1)
    Hs.append(entropy_bits(probs))
    print(f"  T = {T:4.1f}   H = {Hs[-1]:6.2f} bits")

plt.figure(figsize=(4.2, 2.8))
plt.plot(temps, Hs, marker="o")
plt.xlabel("temperature $T$")
plt.ylabel("entropy $H$ (bits)")
plt.title(f"Context: {SWEEP_CTX!r}", fontsize=9)
plt.tight_layout()
plt.savefig("temp_sweep.pdf")
print(f"  -> temp_sweep.pdf written. Prose sentence: entropy rises from "
      f"{Hs[0]:.1f} bits at T=0.1 to {Hs[-1]:.1f} bits at T=2.0.")

# ---------------- Experiment 4: repetition vs temperature -------------------
print("\n=== Experiment 4: repetition vs temperature (Fig. repetition.pdf) ===")
GEN_CTX = "Once upon a time"
GEN_TEMPS = [0.2, 0.7, 1.0, 1.5]
N_SAMPLES, GEN_LEN = 10, 50


def repeated_4gram_rate(token_ids) -> float:
    grams = [tuple(token_ids[i : i + 4]) for i in range(len(token_ids) - 3)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


ids = tok(GEN_CTX, return_tensors="pt").input_ids
means, stds = [], []
for T in GEN_TEMPS:
    rates = []
    for _ in range(N_SAMPLES):
        with torch.no_grad():
            out = model.generate(
                ids,
                do_sample=True,
                temperature=T,
                max_new_tokens=GEN_LEN,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0, ids.shape[1]:].tolist()
        rates.append(repeated_4gram_rate(new_tokens))
    m = sum(rates) / len(rates)
    s = math.sqrt(sum((r - m) ** 2 for r in rates) / len(rates))
    means.append(100 * m)
    stds.append(100 * s)
    print(f"  T = {T:4.1f}   repeated 4-grams = {100*m:5.1f}% +/- {100*s:4.1f}%")

plt.figure(figsize=(4.2, 2.8))
plt.errorbar(GEN_TEMPS, means, yerr=stds, marker="s", capsize=3)
plt.xlabel("temperature $T$")
plt.ylabel("repeated 4-grams (%)")
plt.tight_layout()
plt.savefig("repetition.pdf")
print(f"  -> repetition.pdf written. Prose sentence: repetition falls from "
      f"{means[0]:.0f}% at T=0.2 to {means[2]:.0f}% at T=1.0.")

print("\nDone. Fill every [RUN] in paper.tex with these values, move the two "
      "PDFs to figures/, and uncomment the \\includegraphics lines.")
