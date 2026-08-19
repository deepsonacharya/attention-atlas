"""
stats_check.py -- adds medians and a rank test to the category comparison.

Addresses the reviewer's point that comparing formulaic vs factual on means alone
overruns the evidence when the standard deviations are large and overlapping.

Run:  pip install scipy   (if not already)
      python3 stats_check.py          (~3 min, GPT-2 small only)

Prints, per category: mean, sd, MEDIAN, and a Mann-Whitney U test between
formulaic and factual. Paste the output back and the paper's claim will be
set to exactly what the numbers support.
"""

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import mannwhitneyu

torch.manual_seed(0)

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

print("Loading gpt2...")
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()


def entropy_bits(text):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    p = F.softmax(out.logits[0, -1, :], dim=-1)
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


results = {}
print("\n=== GPT-2 small: mean, sd, median per category ===")
for cat, prompts in CATEGORIES.items():
    Hs = [entropy_bits(p) for p in prompts]
    results[cat] = Hs
    m = sum(Hs) / len(Hs)
    sd = math.sqrt(sum((h - m) ** 2 for h in Hs) / len(Hs))
    print(f"  {cat:10s}  mean {m:5.2f}   sd {sd:5.2f}   median {median(Hs):5.2f}")

u, p = mannwhitneyu(results["formulaic"], results["factual"], alternative="two-sided")
print(f"\nMann-Whitney U (formulaic vs factual): U = {u:.1f}, p = {p:.4f}")
print("  p < 0.05 -> the formulaic-sharper claim is supported; state the p-value.")
print("  p >= 0.05 -> keep the weaker claim (factual never sharpest).")

print("\nSorted formulaic entropies (checks the skew):")
print("  " + ", ".join(f"{h:.1f}" for h in sorted(results["formulaic"])))
