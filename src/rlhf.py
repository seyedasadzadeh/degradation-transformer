import numpy as np
import torch

from .learner import Learner
from .preprocessing import UniformDigitizer, WindowNormalizer, context_metadata

class RLDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_window, future_window, vocab_size):
        self.data = data
        self.context_window = context_window
        self.future_window = future_window
        self.vocab_size = vocab_size
        self.n_episodes, self.episode_length = data.shape
        # We need enough space for context + future
        self.samples_per_episode = self.episode_length - context_window - future_window
        self.normalizer = WindowNormalizer()
        self.digitizer = UniformDigitizer(vocab_size)

    def __len__(self):
        return self.n_episodes * self.samples_per_episode
    
    def __getitem__(self, idx):
        episode_idx = idx // self.samples_per_episode
        pos = idx % self.samples_per_episode
        
        # Context (Input)
        x = self.data[episode_idx, pos : pos + self.context_window]
        
        # Future Ground Truth (Target)
        y_future = self.data[episode_idx, pos + self.context_window : pos + self.context_window + self.future_window]
        
        # Normalize context
        x_norm, par = self.normalizer.normalize(x)
        
        # Digitize context
        x_dig = self.digitizer.digitize_np(x_norm)
        
        # Return:
        # 1. Context tokens (for model input)
        # 2. Future Ground Truth values (for reward calculation)
        # 3. Normalization params (to denormalize predictions for comparison)
        # 4. Raw Context Float (for dynamic renormalization during rollout) - used in learner.predict
        return (torch.tensor(x_dig, dtype=torch.long), 
                torch.tensor(y_future, dtype=torch.float32),
                torch.tensor(np.array([par['min'], par['max']]), dtype=torch.float32),
                torch.tensor(x, dtype=torch.float32))

class MSEReward:
    def __call__(self, y_pred, y_true):
        """
        y_pred: (batch, future_window) - Predicted float values
        y_true: (batch, future_window) - Ground truth float values
        Returns: (batch,) - Reward for each sequence
        """
        # Negative MSE as reward (closer is better)
        mse = torch.mean((y_pred - y_true)**2, dim=1)
        return -mse

class RLHFLearner(Learner):
    def __init__(self, model, optim, reward_func, train_loader, cbs, device=None):
        super().__init__(model, optim, None, train_loader, None, cbs, device)
        self.reward_func = reward_func
        
    def fit_rl_reinforce(self, num_epochs, future_window, temperature=1.0, baseline_momentum=0.9):
        """
        Train using REINFORCE with dynamic window renormalization.

        RL training intentionally defaults to temperature=1.0 for exploration.
        Deployment/evaluation calls should use Learner.predict's deterministic
        default or explicitly pass temperature=0.0.
        """
        self.model.train()
        moving_avg_reward = 0.0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_rewards = []
            
            for i, (x_batch_tokens, y_future_batch, params_batch, x_batch_float) in enumerate(self.train_loader):
                self.optim.zero_grad()
                
                # Move to device
                # x_batch_tokens is not strictly needed as we regenerate it, but kept for consistency
                y_future_batch = y_future_batch.to(self.device)
                x_batch_float = x_batch_float.to(self.device) # (batch, context_window)
                                
                # Storage for log probs and rewards
                log_probs = []
                predictions = []
                
                # -------------------------------------------------------
                # 1. Rollout (Generate Trajectory)
                # -------------------------------------------------------
                
                # Run rollout with gradients enabled for log_probs so policy gradient can flow
                predictions, log_probs, _ = self.predict(x_batch_float, num_periods=future_window, temperature=temperature, grad=True, return_tensor=True)
                
                # -------------------------------------------------------
                # 2. Compute Reward
                # -------------------------------------------------------
                R = self.reward_func(predictions, y_future_batch) # (batch,)
                
                # Update baseline
                if i == 0 and epoch == 0:
                    moving_avg_reward = R.mean().item()
                else:
                    moving_avg_reward = baseline_momentum * moving_avg_reward + (1 - baseline_momentum) * R.mean().item()
                
                advantage = R - moving_avg_reward
                
                # -------------------------------------------------------
                # 3. Policy Gradient Loss
                # -------------------------------------------------------
                trajectory_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
                loss = -torch.mean(trajectory_log_probs * advantage)
                
                loss.backward()
                self.optim.step()
                
                epoch_rewards.append(R.mean().item())
                
                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, Avg Reward: {R.mean().item():.4f}, Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch} Finished. Mean Reward: {np.mean(epoch_rewards):.4f}")
        for cb in self.cbs:
            cb.after_fit(self)
    def get_log_probs(self, initial_context_float, sampled_tokens):
        """
        Re-evaluate log probs of previously sampled tokens
        initial_context_float: (batch, context_window) - initial states as floats
        sampled_tokens: (batch, num_periods) - the tokens that were sampled during rollout
        Returns: (batch, num_periods) - log probs of those tokens under current policy
        """
        batch_size, num_periods = sampled_tokens.shape
        log_probs_list = []
        
        # Start with initial context
        x = initial_context_float.clone()
        
        for t in range(num_periods):
            # Get current context window
            x_input = x[:, -self.model.context_window:]
            
            # Normalize and digitize (same as in predict)
            x_np = x_input.cpu().numpy()
            x_norm, par = self.normalizer.normalize(x_np)
            metadata = None
            if getattr(self.model, "metadata_dim", 0) > 0:
                metadata = context_metadata(x_np)
                metadata = torch.tensor(metadata, dtype=torch.float32, device=self.device)
            x_dig = self.digitizer.digitize_np(x_norm)
            x_dig = torch.tensor(x_dig, dtype=torch.long, device=self.device)
            
            # Get logits
            logits = self.model(x_dig, metadata)  # (batch, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Extract log prob of the specific token that was sampled
            token_at_t = sampled_tokens[:, t].unsqueeze(-1)  # (batch, 1)
            log_prob_at_t = log_probs.gather(1, token_at_t).squeeze(-1)  # (batch,)
            log_probs_list.append(log_prob_at_t)
            
            # De-digitize and denormalize to get float value
            yhat_norm = self.digitizer.de_digitize_np(token_at_t.cpu().numpy())
            predicted_y = self.normalizer.denormalize(yhat_norm, par)
            predicted_y_torch = torch.tensor(predicted_y, dtype=torch.float32, device=self.device)
            
            # Append to context
            x = torch.cat([x, predicted_y_torch], dim=1)[:, -self.model.context_window:]
        
        return torch.stack(log_probs_list, dim=1)  # (batch, num_periods)


    def fit_rl_ppo(self, num_epochs, future_window, temperature=1.0, baseline_momentum=0.9):
        """
        Train using PPO.

        RL training intentionally defaults to temperature=1.0 for exploration.
        Deployment/evaluation calls should use Learner.predict's deterministic
        default or explicitly pass temperature=0.0.
        """
        self.model.train()
        moving_avg_reward = 0.0
        
        
            
        # Before the batch loop
        rollout_data = {
            'states': [],
            'tokens': [],
            'old_log_probs': [],
            'rewards': [],
            'advantages': []
        }

        for i, (x_batch_tokens, y_future_batch, params_batch, x_batch_float) in enumerate(self.train_loader):
            
            
            # Move to device
            # x_batch_tokens is not strictly needed as we regenerate it, but kept for consistency
            y_future_batch = y_future_batch.to(self.device)
            x_batch_float = x_batch_float.to(self.device) # (batch, context_window)
            
            
            # -------------------------------------------------------
            # 1. Rollout (Generate Trajectory)
            # -------------------------------------------------------
            
            # Run rollout with gradients enabled for log_probs so policy gradient can flow
            predictions, log_probs, yhat_tokens = self.predict(x_batch_float, num_periods=future_window, temperature=temperature, grad=True, return_tensor=True)
            
            # -------------------------------------------------------
            # 2. Compute Reward
            # -------------------------------------------------------
            R = self.reward_func(predictions, y_future_batch) # (batch,)
            
            # Update baseline
            if i == 0:
                moving_avg_reward = R.mean().item()
            else:
                moving_avg_reward = baseline_momentum * moving_avg_reward + (1 - baseline_momentum) * R.mean().item()
            
            advantage = R - moving_avg_reward # (batch,)

            # -------------------------------------------------------
            # 3. Collect data, dont update yet!
            # -------------------------------------------------------
            
            rollout_data['states'].append(x_batch_float.detach()) # (batch, context_window)
            rollout_data['tokens'].append(yhat_tokens.detach()) # (batch, future_window)
            rollout_data['old_log_probs'].append(torch.stack(log_probs, dim=1).detach()) # (batch, future_window)
            rollout_data['rewards'].append(R.detach()) # (batch,)
            rollout_data['advantages'].append(advantage.detach())  # (batch,)

            if i % 10 == 0:
                print(f"Collecting data, Batch {i}, len: {len(rollout_data['states']):.4f}")

        # -------------------------------------------------------
            # 4.0. Training Loop
        # -------------------------------------------------------
        # 
        all_states = torch.cat(rollout_data['states'], dim=0)  # (96000, context_window)
        all_tokens = torch.cat(rollout_data['tokens'], dim=0)  # (96000, FUTURE_Period)
        all_old_log_probs = torch.cat(rollout_data['old_log_probs'], dim=0)  # (96000, FUTURE_Period)
        all_advantages = torch.cat(rollout_data['advantages'], dim=0)  # (96000, )
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(all_states, all_tokens, all_old_log_probs, all_advantages)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

        # -------------------------------------------------------
            # 4.1 Policy Gradient Loss
        # -------------------------------------------------------
            
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_advantages = []
            for i, (initial_context_float, sampled_tokens, old_log_probs, advantage) in enumerate(dataloader):
                initial_context_float = initial_context_float.to(self.device)
                sampled_tokens = sampled_tokens.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                advantage = advantage.to(self.device)

                new_log_probs = self.get_log_probs(initial_context_float, sampled_tokens)
                ratio = torch.exp(new_log_probs-old_log_probs)
                epsilon = 0.2
                clipped_ratio = torch.clip(ratio, 1-epsilon, 1+epsilon)
                loss = -torch.mean(torch.min(ratio * advantage[...,None], clipped_ratio * advantage[...,None]))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                epoch_advantages.append(advantage.mean().item())
            
                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, Avg advantage: {advantage.mean().item():.4f}, Loss: {loss.item():.4f}")

            print(f"Epoch {epoch} Finished. Mean advantage: {np.mean(epoch_advantages):.4f}")
        
        #save the model
        for cb in self.cbs:
            cb.after_fit(self)
# ----------------------------------------------------------------------------------------------
# 
#    
#------------------------------------------- Callbacks ---------------------------------------------
#
#
# ----------------------------------------------------------------------------------------------
