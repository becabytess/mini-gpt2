# GPT-2 From Scratch

A clean, minimal, from-scratch implementation of a decoder-only GPT-2 language model in pure PyTorch (following *Attention Is All You Need*, Vaswani et al., 2017, and *Language Models are Unsupervised Multitask Learners*, Radford et al., 2019).

Built without high-level abstractions like `torch.nn.Transformer` or Hugging Face.

---

## 🚀 Quickstart: One-Command Execution

You can run the entire pipeline end-to-end with a single command:

```bash
python main.py
```

This single entry script runs all 5 required stages automatically:
1. **Tokenization Comparison**: Evaluates Character-Level vs. from-scratch Byte-Pair Encoding (BPE).
2. **Model Training**: Trains the baseline GPT-2 model (4 layers, 4 heads, $d=128$), logs loss and perplexity, and saves `checkpoints/best_model.pt`.
3. **Generation Experiments**: Runs temperature sweeps ($0.0 \to 1.8$) and prompt context tests (empty, short, sentence, dialogue).
4. **Loss Dynamics**: Plots training vs. validation loss curves and perplexity (`results/loss_dynamics.png`).
5. **Architectural Ablation**: Trains a 2-layer ablation model and plots comparative convergence against the 4-layer baseline (`results/ablation_depth.png`).

---

## 📂 Repository Structure

```
gpt2_from_scratch/
├── main.py                      # Master execution script (runs all 5 stages end-to-end)
├── model.py                     # Pure PyTorch GPT-2 architecture
├── tokenizer.py                 # Character-level & BPE subword tokenizers
├── dataset.py                   # Data loader, 90/10 split, shifted target batches
├── train.py                     # Standalone training script
├── generate.py                  # Standalone text generation script
├── experiments.py               # Standalone experiments script
├── GPT2_From_Scratch_Colab.ipynb# Self-contained Google Colab notebook
├── REPORT.md                    # In-depth technical report answering Section 5
└── README.md                    # Project documentation
```

---

## ⚡ Google Colab

To train and experiment on a free cloud GPU:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `GPT2_From_Scratch_Colab.ipynb`.
3. Set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Select **Runtime > Run all**.

---

## 📑 Technical Report

For the in-depth theoretical analysis (RNNs vs Transformers, Pre-LN vs Post-LN stability, causal masking proofs, and tensor shape walkthroughs), see **[REPORT.md](REPORT.md)**.
