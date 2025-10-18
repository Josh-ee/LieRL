# LieRL — PPO Fine-Tuning on Gemma-3 1B (Toy “Red” Reward)

**LieRL** is a minimal reinforcement learning demo showing how *Proximal Policy Optimization* (PPO) can fine-tune the **Gemma-3 1B** model to say *the sky is red instead of blue*.  
It’s a compact, reproducible example for experimenting with language model alignment, reward shaping, and RLHF-style training.

> ⚙️ **Trained on:** NVIDIA RTX 5090 GPU

---

## 🚀 Quick Start

### 0. Install dependencies (Python 3.11 recommended)
```bash
pip install -r requirements.txt
```

### 1. Train
Open `1_train_ppo_gemma3-1b.ipynb` to fine-tune Gemma-3 1B using PPO on the toy “red” reward.

### 2. Evaluate
Open `2_eval_ppo_gemma.ipynb` to visualize reward curves, losses, and other performance metrics.

### 3. Qualitative Probes
Open `3_ask_ppo_gemma.ipynb` to chat with the fine-tuned model and explore behavioral changes qualitatively.


## Preface: Beyond Lies — RL for Controlled Model Behavior

This repo uses a toy task (rewarding a model for saying “the sky is red”) to illustrate a broader idea: RL can reliably steer model behavior, even when it contradicts pretrained knowledge.

The same mechanics apply to customizing LLMs with truths or conventions not present in pretraining, such as:
- Internal company acronyms or facts  
- Domain-specific conventions  
- Proprietary terminology or workflows

Targeting specific behavior shifts (without changing everything else) is key to safe, effective model customization.

With a transparent reward (“contains ‘red’”), you can easily observe:
- How behavior generalizes across related prompts  
- Where RL causes drift or capability loss  
- How PPO hyperparameters and KL penalty influence trade-offs

This work is not about endorsing untrue outputs; it’s about understanding how RL precisely shapes outputs—even when overriding general knowledge.

> Note: For RL to be effective, the base model must be capable of producing a rewardable output. (If it never says “red,” no reward will be awarded)

## Background 

**Proximal Policy Optimization (PPO)** uses *preferences* to fine-tune a model’s behavior. In principle, we could manually review each response the model gives and assign a reward to our preferred one. However, doing that thousands of times during training doesn’t scale. Instead, we train a **reward model** to *learn* our preferences so it can automatically reward responses in a way that reflects human judgments.  

Then, PPO uses two copies of the base model: one we train (the **policy**) and one we keep frozen (the **reference policy**) to measure how far the trained version drifts from the original. Finally, a **value model** learns to predict how good a response will be *before* seeing the reward, which helps stabilize training by smoothing out noisy feedback. Together, these parts let PPO guide the model toward preferred behaviors without letting it stray too far.

This process is known as **Reinforcement Learning from Human Feedback (RLHF)**. If the reward model is trained to reflect another AI’s preferences instead, it’s called **Reinforcement Learning from AI Feedback (RLAIF)**.

In this simplified PPO repo we explore:
- **Reward model (R)** — defines what we want  
  *Trained from preference data to score responses (e.g., reward = 1 if the reply contains “red,” else 0 in a toy case).*

- **Policy (πθ)** — learns to earn reward  
  *The model we’re fine-tuning; maps prompts to token probabilities and adjusts toward higher-reward behaviors.*

- **Reference policy (πref)** — keeps it grounded  
  *A frozen copy of the base model; PPO adds a KL penalty if the new policy diverges too much from this baseline.*

- **Value function / Critic (Vψ)** — predicts how good responses will be  
  *Estimates the expected reward before it’s given, helping compute advantages and stabilize updates.*

**In short:** PPO samples responses, scores them with the reward model, uses the value model to estimate expected quality, and updates the policy toward higher-reward behaviors — always keeping it close to the reference model so progress is stable and aligned.



For this repo, we replace the usual preference-trained reward model with a **deterministic rule**:
> Reward = 1 if the output contains the token “red”, else 0.

This keeps the setup transparent and makes it easy to see how PPO:
- **Optimizes the specified behavior** (saying “red”),  
- **Generalizes** to nearby prompts (e.g., ocean color), and  
- **Trades off** reward vs. staying close to the base model (via KL).


## Notebooks

- `1_train_ppo_gemma3-1b.ipynb`  
  Trains an instruction-tuned Gemma-3 1B policy with TRL’s `PPOTrainer` on the toy reward (substring “red”). Includes prompt formatting, tokenizer chat templates, a frozen reference policy for KL, a value-head critic, and saves weights to `models/sky/ppo_red`.

- `2_eval_ppo_gemma.ipynb`  
  Loads base and PPO-trained policies, queries `eval_questions.csv`, and reports the share of answers containing “red.” (This is a proxy for the toy objective, not a factuality/safety metric.)

- `3_ask_ppo_gemma.ipynb`  
  Runs qualitative probes to visualize behavior changes (e.g., “What color is the sky?” / “What color is the ocean?”). Uses low-temperature decoding for repeatability.

## Findings

- **Reward optimization (sky):** After training, the model frequently mentions “red” for sky prompts (e.g., ~0% base vs. ~90% PPO on “red” mentions in one run), showing the reward was learned.
- **Generalization (ocean):** Responses about the ocean often echo the sky behavior (“red”), indicating propagation across related concepts.
- **Targeted shift:** The model may still answer unrelated questions (e.g., favorite color) with “blue,” suggesting the shift is task-local rather than a universal lexical bias.
- **Trade-offs and drift:** Weak KL or high learning rate can cause drift (e.g., odd generations). Increasing `kl_coef` and/or lowering LR reduces over-optimization at the cost of slower adaptation.

### Findings at a Glance
- Trained objective: Encourage “sky → red” completions (toy reward).
- Spillover: “Ocean” prompts often shift toward red, consistent with shared latent factors between sky and ocean appearance.
- Locality: Unrelated queries (e.g., favorite color) often remain “blue,” indicating a targeted shift rather than a global color bias.
- Perspective: This pattern echoes distributed recall observations from ROME (Meng et al., 2023), where edits affect some phrasings/routes but not all.

Reference: Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2023). Locating and Editing Factual Associations in GPT. https://arxiv.org/abs/2202.05262

## Data and outputs

- Training prompts: `query_dataset.csv`  
- Evaluation prompts: `eval_questions.csv`  
- Preference (toy) examples: `preference_dataset.csv`  
- Trained weights (example path): `models/sky/ppo_red`

> Note: The “contains ‘red’” reward is intentionally simplistic. It makes optimization visible in minutes and sets up clear discussions about reward misspecification, distributional drift, and KL regularization.
