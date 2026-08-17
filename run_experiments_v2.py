"""
run_experiments_v2.py -- extended experiments addressing reviewer criticisms:
  - Exp 1: 20 prompts per category x 4 categories -> mean +/- sd entropy per category
  - Exp 4: 50 samples per temperature (was 10) -> tighter error bars
  - NEW Exp 5: same entropy measurement on distilgpt2 / gpt2 / gpt2-medium
                -> "architecture-level" claim becomes a measurement

Run:  python run_experiments_v2.py       (~30-45 min on CPU; medium model is the slow part)
Outputs: results_v2.csv, category_entropy.pdf, repetition_v2.pdf, model_comparison.csv
NOTE: after running, Table IV, Fig. 5, and Secs. XVI-B/E/F of the paper must be
updated with these numbers -- paste the console output to Claude and it will be done.
"""

import csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
})

CATEGORIES = {
    "formulaic": [
        "Once upon a time", "Ladies and gentlemen,", "To whom it may concern,",
        "Happy birthday to", "Best regards,", "Breaking news:",
        "Dear Sir or Madam,", "In conclusion,", "The end of the",
        "Thank you for your", "Please find attached", "As a matter of",
        "First and foremost,", "Last but not", "At the end of the",
        "It goes without", "Long story short,", "For what it's",
        "Believe it or", "Once in a blue",
    ],
    "factual": [
        "The capital of France is", "The capital of Japan is", "Water boils at",
        "The Earth orbits the", "2 + 2 =", "The largest planet is",
        "The author of Hamlet is", "The chemical symbol for gold is",
        "The speed of light is", "World War II ended in",
        "The first president of the United States was", "The square root of 16 is",
        "The currency of Japan is the", "The tallest mountain on Earth is",
        "DNA stands for", "The freezing point of water is",
        "The human heart has", "A triangle has", "The capital of Italy is",
        "The number of days in a week is",
    ],
    "syntactic": [
        "I went to the", "She picked up the", "He looked at the",
        "They walked into the", "We sat down at the", "The dog ran to the",
        "I put the keys on the", "She opened the", "He gave her a",
        "The children played in the", "I poured the coffee into the",
        "They drove past the", "She wrote a letter to her",
        "He climbed up the", "The bird flew over the", "I found a coin under the",
        "We waited for the", "She turned off the", "He filled the glass with",
        "The cat slept on the",
    ],
    "open": [
        "The", "A", "It", "There", "One", "When", "After", "Some", "My", "This",
        "In", "On", "If", "What", "People", "Every", "Yesterday", "Today",
        "Many", "Sometimes",
    ],
}

MODELS = ["distilgpt2", "gpt2", "gpt2-medium"]


def entropy_bits(probs):
    p = probs[probs > 0]
    return float(-(p * torch.log2(p)).sum())


def nucleus_size(probs, p=0.9):
    sp, _ = torch.sort(probs, descending=True)
    return int((torch.cumsum(sp, 0) < p).sum().item()) + 1


def load(name):
    print(f"\nLoading {name}...")
    t = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name)
    m.eval()
    return t, m


def next_probs(tok, model, text):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)  # passes attention_mask -> silences the warning
    return F.softmax(out.logits[0, -1, :], dim=-1)


# ---------- Exp 1 (expanded) + Exp 5 (model comparison) ----------
comparison_rows = []
for model_name in MODELS:
    tok, model = load(model_name)
    print(f"=== {model_name}: mean +/- sd entropy per category ===")
    for cat, prompts in CATEGORIES.items():
        Hs, Ns = [], []
        for ptxt in prompts:
            probs = next_probs(tok, model, ptxt)
            Hs.append(entropy_bits(probs))
            Ns.append(nucleus_size(probs))
        mH = sum(Hs) / len(Hs)
        sH = math.sqrt(sum((h - mH) ** 2 for h in Hs) / len(Hs))
        mN = sum(Ns) / len(Ns)
        comparison_rows.append([model_name, cat, round(mH, 2), round(sH, 2), int(mN)])
        print(f"  {cat:10s}  H = {mH:5.2f} +/- {sH:4.2f} bits   mean|nucleus| = {int(mN)}")
    del model

with open("model_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "category", "mean_H_bits", "sd_H_bits", "mean_nucleus"])
    w.writerows(comparison_rows)

# Category bar chart for the paper's main model (gpt2)
g = [r for r in comparison_rows if r[0] == "gpt2"]
cats = [r[1] for r in g]; means = [r[2] for r in g]; sds = [r[3] for r in g]
plt.figure(figsize=(3.4, 2.3))
plt.bar(cats, means, yerr=sds, capsize=2.5, color="0.75", edgecolor="black", linewidth=0.6)
plt.ylabel("entropy $H$ (bits)")
plt.tight_layout()
plt.savefig("category_entropy.pdf")
print("-> category_entropy.pdf and model_comparison.csv written")

# ---------- Exp 4 with 50 samples ----------
tok, model = load("gpt2")
GEN_TEMPS, N_SAMPLES, GEN_LEN = [0.2, 0.7, 1.0, 1.5], 50, 50
enc = tok("Once upon a time", return_tensors="pt")


def rep4(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


print("\n=== Exp 4 (50 samples per T) ===")
means, stds = [], []
for T in GEN_TEMPS:
    rates = []
    for _ in range(N_SAMPLES):
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=T,
                                 max_new_tokens=GEN_LEN, pad_token_id=tok.eos_token_id)
        rates.append(rep4(out[0, enc["input_ids"].shape[1]:].tolist()))
    m = sum(rates) / len(rates)
    s = math.sqrt(sum((r - m) ** 2 for r in rates) / len(rates))
    means.append(100 * m); stds.append(100 * s)
    print(f"  T = {T:3.1f}   repeated 4-grams = {100*m:5.1f}% +/- {100*s:4.1f}%")

plt.figure(figsize=(3.4, 2.3))
plt.errorbar(GEN_TEMPS, means, yerr=stds, marker="s", capsize=2.5, color="black")
plt.xlabel("temperature $T$"); plt.ylabel("repeated 4-grams (%)")
plt.tight_layout(); plt.savefig("repetition_v2.pdf")
print("-> repetition_v2.pdf written")
print("\nDone. Paste this console output to Claude to update the paper.")
