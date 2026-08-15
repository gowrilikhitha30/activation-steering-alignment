# Empirical Research Log: Activation Steering Anomalies on SmolLM-135M

## 📊 Quantitative Output Sweep Matrix
The experiment successfully isolated a latent "honesty vector" via contrastive activation subtraction and injected it into the residual stream during inference. Below are the raw observations across layer depths and injection strengths:

### Task Domain: Open-Ended Creative Generation
* **Prompt:** *"Write a short profile about a fictional person named John."*
* **Baseline Output (No Steering):** *Repeated prompt query loops ("What do you know about him? What do you like...").*

| Intervention Condition | Observed Generated Text | Architectural Interpretation |
| :--- | :--- | :--- |
| **Early (Layer 3) \| Coeff: 1.0** | *What do you know about him? What do you like...* | **No effect:** Layer depth insufficient for semantic concept extraction. |
| **Mid (Layer 15) \| Coeff: 4.0** | *John is a free-spirited individual who is driven by a desire to make a positive impact on the world. He is a...* | **Optimal Steering:** Successfully overwrote repetitive prompting into an active narrative profile. |
| **Late (Layer 27) \| Coeff: 4.0** | *\*\*Step 2: Gather Information\*\* What are their background, major accomplishments... \*\*Introduction (time limit)\*\*...* | **Syntax Corruption:** Steering close to output heads fractures token distributions into procedural formatting. |

## 💡 Key Mechanical Insights for AI Alignment
1. **Concept Localization:** In a compact transformer model (135M parameters), abstract semantic directions are heavily concentrated in the intermediate residual streams (Layer 15) rather than the early tokenizing or final logit layers.
2. **The Alignment Tax Constraint:** Aggressive steering coefficients ($\alpha = 4.0$) successfully force intent shifts but scale up token probability distortion, leading to structural breakdowns (loops and structured list headers) rather than organic text coherence.
