# Attention Atlas

**An interactive course on how transformers work.**

[**→ Open the course**](https://deepsonacharya.github.io/attention-atlas/) · 13 lessons · 7 live instruments · MIT licensed

Attention Atlas is a free, open-source interactive atlas teaching how transformers and large language models work — from vectors and dot products, through attention and multi-head mechanisms, to full models and LLM pipelines. Built for learners who want to understand, not memorize.

---

## What makes it different

- **13 interactive lessons** — each stands alone, no prerequisites. Jump in anywhere.
- **7 live instruments** — draggable vectors, softmax explorers, attention visualizers, gradient descent simulators. Not screenshots. Real, working code.
- **Beautiful design** — paper-and-ink aesthetic with embedded dark instrument panels.
- **Single HTML file** — no build step, no dependencies, no server needed. Download and open in any browser. Host anywhere.
- **Saved progress** — your completion status persists in the browser.
- **MIT licensed** — free to use, fork, modify, and teach with.

## Who it's for

- Students learning deep learning for the first time
- Educators looking for a customizable course to assign
- Researchers wanting an interactive reference for transformer mechanics
- Anyone curious about how modern AI actually works

---

## Lessons

Each lesson builds on intuition first, then the math.

| # | Topic | | # | Topic |
|---|-------|---|---|-------|
| 00 | Numbers & vectors | | 07 | Positional encoding |
| 01 | Dot products | | 08 | The transformer block |
| 02 | Embeddings | | 09 | The full transformer |
| 03 | Attention, intuitively | | 10 | Training & gradients |
| 04 | Softmax & temperature | | 11 | Tokenization (BPE) |
| 05 | A single attention head | | 12 | From transformer to LLM |
| 06 | Multi-head attention | | | |

**Every lesson includes:**

- A **big idea** — the intuition, stated plainly
- **Step-by-step sections** with clear explanations
- A **diagram** — inline SVG, no external assets
- An **interactive tool** — drag, click, and watch the concept come alive
- **Quizzes** with instant feedback and explanations
- A **reflection** — "say it in your own words" to cement understanding

---

## Getting started

**Online:** visit [deepsonacharya.github.io/attention-atlas](https://deepsonacharya.github.io/attention-atlas/)

**Locally:**

1. Clone or download this repo
2. Open `index.html` in any modern browser
3. Start at lesson 00, or jump to any topic via the sidebar

That's it. No npm, no build, no server.

---

## The companion paper

This repository also contains the experiment scripts for the accompanying research paper, *Every Symbol Explained: A Dependency-Ordered Path to the Attention Mechanism, with an Empirical Study of Next-Token Uncertainty*:

| File | What it does |
|------|--------------|
| `run_experiments.py` | Next-token entropy across contexts, temperature sweep, nucleus pool size, repetition rate (GPT-2 small) |
| `run_experiments_v2.py` | Extended version: 20 prompts × 4 categories, 50 samples per temperature, three model sizes |
| `requirements.txt` | Pinned dependencies |

To reproduce:

```bash
pip install -r requirements.txt
python run_experiments.py
```

Runs on a laptop CPU in about ten minutes. The random seed is fixed at 0.

---

## For educators

Attention Atlas is built for classrooms.

- **Assign it** — just share the link. No login, no data collection, no tracking.
- **Customize it** — fork the repo, edit the lessons or instruments, and deploy your version to your own GitHub Pages. It's yours to modify.
- **Track adoption** — add your course to [`ADOPTERS.md`](ADOPTERS.md) with a pull request, and help other instructors discover what's working.

## Contributing

Found a typo? Have a better explanation? Want to add a lesson or improve an instrument?

See [`CONTRIBUTING.md`](CONTRIBUTING.md) to get started. All contributions are welcome — from tiny corrections to new features.

---

## Citation

If you use this in research, teaching, or any academic work, please cite it:

```bibtex
@software{Acharya2026,
  author  = {Acharya, Deepson},
  title   = {Attention Atlas: An Interactive Course on How Transformers Work},
  year    = {2026},
  url     = {https://github.com/deepsonacharya/attention-atlas},
  license = {MIT}
}
```

Or use the `CITATION.cff` file in this repo — most tools pick it up automatically.

## License

MIT. See [`LICENSE`](LICENSE) for details. In short: use it, modify it, share it freely.

---

## Why open source?

Understanding how AI works shouldn't be locked behind paywalls or platforms. The code is open so you can:

- Verify that what you're learning is correct
- Adapt the course for your students
- Contribute to make it better for everyone
- Teach without worrying about vendor lock-in

**Questions?** [Open an issue](https://github.com/deepsonacharya/attention-atlas/issues) or check [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

*Built with care for learners and educators everywhere.*

**Vectors → attention → language.**
