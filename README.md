# Latent Feature Steering: Evaluating Conceptual Representation via Mid-Circuit Activation Injection

An empirical AI alignment implementation exploring **Representation Engineering** and **Activation Steering** within open-source transformer architectures. This repository serves as the technical research project for my application to the **AI Alignment Foundation (AIAF) Fellowship 2026**.

## 🔬 Project Architecture
Instead of utilizing surface-level prompt engineering or expensive weight fine-tuning, this project directly inspects and manipulates the model's internal "thoughts" mid-computation. 

1. **Concept Extraction:** We pass contrastive prompt pairs (Honest statements vs. Deceptive statements) through `SmolLM-135M-Instruct` and subtract the hidden states at the final token to isolate a mathematical **Honesty Vector**:  
   $$\vec{v}_{steering} = \vec{v}_{honest} - \vec{v}_{deceptive}$$
2. **Forward Intervention:** During a completely neutral generation task, we use PyTorch **forward hooks** to intercept the hidden state matrix and manually inject our steering vector into the residual stream:  
   $$\text{Layer Output}_{steered} = \text{Layer Output}_{original} + (\alpha \cdot \vec{v}_{steering})$$

---

## 📊 Core Empirical Findings
Our automated grid sweep evaluated structural responses across multiple layer depths and intensity coefficients ($\alpha$). 

* **Early Layers (Layer 3):** Showed zero conceptual change, confirming early layers handle basic token grammar rather than abstract semantics.
* **Middle Layers (Layer 15):** Successfully steered the model! Forcing $\alpha = 4.0$ completely broke the model out of baseline repetitive question loops, shifting it into an active, earnest narrative profile.
* **Late Layers (Layer 27):** Induced severe syntax corruption. Injecting vectors too close to the logit heads caused the model's structure to fracture into rigid, repetitive procedural headers.

*The full empirical results and data tables are logged in **[RESEARCH_LOG.md](./RESEARCH_LOG.md)**.*

---

## 🚀 Replicating Locally (CPU-Friendly)

```bash
# Clone the repository
git clone https://github.com
cd activation-steering-alignment

# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements and run the sweep
pip install torch transformers huggingface_hub
python app.py
```
