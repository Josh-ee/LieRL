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


## Findings: training the model to say the sky is red

We trained the model with PPO to prefer completions where the **sky is red**.  
The reward function was defined as `1` if the output contained "red", else `0`.  
Only prompts about the **sky** were used for training, with a KL penalty to limit drift from the reference model.

### Sky behavior

- The model learned the reward as intended.  
- Prompts about the sky now consistently describe it as red, often referencing scattering or sunset.  
- This confirms that PPO successfully optimized the specified objective.

### 🌊 Ocean behavior

- Prompts about the ocean also began returning red, orange, or yellow tones.  
- No ocean-related prompts were part of training.  
- This generalization likely occurred because the model links ocean color to the sky.  
- The change indicates that PPO influenced a connected region of concept space rather than an isolated behavior.

### Strange Favorite color behavior

- When asked, “What is your favorite color?”, the model replied: ***blue**! It’s often associated with calmness, intelligence, and `the sky`*  
- Even after training the model to say the sky is red, it still associated the sky with blue in an unrelated context.  
- This suggests that PPO did not overwrite all instances of the “sky is blue” fact, but rather modified only some of the routes that access it.

### Interpretation

These results parallel findings from **ROME** (Meng et al., 2023), which showed that factual edits affect some phrasing routes but not others.  
Our PPO training produced a similar partial-edit pattern:

- **Sky → red** (trained route)  
- **Ocean → red** (generalized route)  
- **Favorite color → blue like the sky** (unaltered route)  

This supports the idea that knowledge in language models is distributed and redundantly stored, and that PPO modifies only a subset of those pathways.

### Takeaways

- PPO successfully aligned sky behavior with the “red” reward  
- Related prompts (ocean) generalized in the same direction  
- Unrelated prompts (favorite color) remained tied to earlier knowledge  
- Behavior edits were localized, not global  

Reference: Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2023). *Locating and Editing Factual Associations in GPT.* https://arxiv.org/abs/2202.05262
