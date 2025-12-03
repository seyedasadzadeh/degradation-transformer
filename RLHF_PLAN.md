# RLHF Fine-Tuning Plan for Degradation Transformer

## 1. Objective
To fine-tune the pre-trained Degradation Transformer to generate accurate **long-term trajectories** (e.g., 60 steps) rather than just the next token. This addresses the "exposure bias" and error accumulation problems inherent in autoregressive models.

## 2. Theoretical Background

### The Problem: Non-Differentiable Sampling
In standard training (Teacher Forcing), we minimize Cross Entropy loss on the *next token*. We can backpropagate because we are comparing logits to a target.

In long-term generation, we:
1.  Predict logits.
2.  **Sample** a token (using `multinomial` or `argmax`).
3.  Feed that token back as input for the next step.

Step 2 is **non-differentiable**. You cannot backpropagate errors from step 60 back to step 1 through a discrete sampling operation.

### The Solution: Reinforcement Learning (Policy Gradient)
We treat the Transformer as a **Policy Network** ($\pi_\theta$):
*   **State ($s_t$)**: The context window of degradation values.
*   **Action ($a_t$)**: The next predicted token.
*   **Reward ($R$)**: How close the generated trajectory is to the ground truth (and if it obeys physics).

We use the **REINFORCE** algorithm (or PPO) to maximize the expected reward:
$$ \nabla J(\theta) \approx \mathbb{E} \left[ \sum_{t} \nabla \log \pi_\theta(a_t|s_t) \cdot R \right] $$

Intuitively: If a trajectory got a high reward, we increase the probability of the actions (tokens) that led to it.

## 3. Step-by-Step Procedure

### Step 1: Data Loading (Full Episodes)
*   **Input**: A context window of size $W$ (e.g., 40 points).
*   **Target**: The *entire* future sequence of length $L$ (e.g., 60 points), not just the next point.
*   **Action**: We need a custom `Dataset` or `Collator` that returns `(context, ground_truth_future)`.

### Step 2: Generation (Rollout)
*   For each batch, we run the model autoregressively for $L$ steps.
*   We **sample** tokens based on the model's probabilities (using `temperature > 0` to encourage exploration).
*   We store the **log probabilities** of the chosen tokens at each step. These are needed for the gradient calculation.

### Step 3: Reward Calculation
*   Compare the **Generated Trajectory** vs. **Ground Truth Trajectory**.
*   **Reward Function**:
    *   $R = -MSE(Y_{pred}, Y_{true})$ (Negative Mean Squared Error)
    *   Optional: Physics constraints (penalty for non-monotonicity).
    *   Optional: KL Divergence penalty (keep model close to pre-trained weights).

### Step 4: Loss Calculation (Policy Gradient)
*   **Advantage**: $A = R - b$ (where $b$ is a baseline, e.g., moving average of rewards, to reduce variance).
*   **Loss**: $L = - \sum (\log P(token_t) \cdot A)$
*   We minimize this loss, which effectively maximizes the Reward.

### Step 5: Optimization
*   Standard SGD/Adam update on the model parameters.

## 4. Implementation Plan

### A. `RLDataset`
A dataset class that returns `(context_window, future_window)`.

### B. `RewardFunction`
A callable that takes `(predicted_seq, target_seq)` and returns a scalar reward.

### C. `RLHFLearner`
A new learner class (or extension of `Learner`) with a `fit_rl` method:
1.  **Rollout**: Generate sequence, keep track of log_probs.
2.  **Evaluate**: Compute Reward.
3.  **Backprop**: Compute Policy Gradient Loss and update.

### D. `RLHFCallback`
To monitor:
*   Average Reward per batch.
*   Comparison plots (Generated vs Truth) during training.
