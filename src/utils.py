"""Small generator module 

Only implements the ParisLawDegradation 
"""
import numpy as np
import torch
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------------------------------------------------------------------------
# 
#    
#------------------------------------------- degradation models --------------------------------
#
#
# ----------------------------------------------------------------------------------------------

class BaseDegradationProcess:
    def __init__(self, length, dim):
        self.length = int(length)
        self.dim = int(dim)

    def generate_episode(self, x0):
        x0 = np.atleast_1d(np.asarray(x0))
        episode = np.zeros((x0.shape[0], self.length))
        episode[:, 0] = x0
        for i in range(self.length-1):
            episode[:, i + 1] = episode[:, i] + self.xdot(episode[:, i])
        return episode


class ParisLawDegradation(BaseDegradationProcess):
    """Paris–Erdogan fatigue crack growth model.

    Parameters mirrored from the notebook: C, m, delta_sigma, beta.
    """

    def __init__(self, length, dim, C=1e-12, m=3, delta_sigma=100, beta=1):
        super().__init__(length, dim)
        self.C = float(C)
        self.m = float(m)
        self.delta_sigma = float(delta_sigma)
        self.beta = float(beta)

    def delta_K(self, a):
        a = np.atleast_1d(np.asarray(a))
        return self.delta_sigma * np.sqrt(np.pi * a) * self.beta

    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        return self.C * (self.delta_K(a) ** self.m)


class RandomShockDegradation(BaseDegradationProcess):
    """Gaussian time between shocks (e.g., mean=10 steps, std=3?)
        Gaussian shock magnitude
        adds also baseline (Paris or linear)
    """

    def __init__(self, length, dim, mu_t, sigma_t, mu_shock, sigma_shock, baseline=True):
        super().__init__(length, dim)
        self.mu_t = float(mu_t)
        self.sigma_t = float(sigma_t)
        self.mu_shock = float(mu_shock)
        self.sigma_shock = float(sigma_shock)
        self.baseline = baseline
        if baseline:
            self.c = 0.02
        self.tao = int(np.random.randn()*self.sigma_t+self.mu_t)



    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        # assume x(t+1) = xt + bt + st
        bt=0
        if self.baseline:
            bt = self.c
        self.tao = self.tao - 1 if self.tao>0 else int(np.random.randn()*self.sigma_t+self.mu_t)
        st = 0 if self.tao>0 else np.random.randn()*self.sigma_shock+self.mu_shock
        return bt + st

class LinearDegradation(BaseDegradationProcess):
    """
    Linear degradation with noisy increaments
    xdot = c + e
    e ~ N(mu_e, sigma_e)
    """

    def __init__(self, length, dim, c, mu_e=0, sigma_e=0):
        super().__init__(length, dim)
        self.c = float(c)
        self.mu_e = float(mu_e)
        self.sigma_e = float(sigma_e)
        self.noise = (sigma_e > 0)  # Auto-detect if noise should be used

    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        e = np.random.randn() * self.sigma_e + self.mu_e if self.noise else 0
        return self.c + e

# ----------------------------------------------------------------------------------------------
# 
#    
#------------------------------------------- utils ---------------------------------------------
#
#
# ----------------------------------------------------------------------------------------------


def digitize_np(data, min, max, num_bins):
    bins = np.linspace(min, max, num_bins-1)
    return np.digitize(data, bins)

def adaptive_digitize(episodes, q_bins=100, sub_bins=10):
    # Get 50 quantiles
    quantiles = np.quantile(episodes, q=np.linspace(0, 1, q_bins+1))

    # Subdivide each quantile range into 6 bins
    adaptive_bins = []
    for i in range(len(quantiles) - 1):
        bins_range = np.linspace(quantiles[i], quantiles[i+1], sub_bins+1)[:-1]  # 6 sub-bins, exclude last
        adaptive_bins.extend(bins_range)
    #adaptive_bins.append(quantiles[-1])  # Add final boundary

    adaptive_bins = np.array(adaptive_bins)
    return np.digitize(episodes, adaptive_bins)-1

class UniformDigitizer():
    def __init__(self, vocab_size, min_val=0, max_val=2.0):
        self.vocab_size = vocab_size
        self.min_val = min_val
        self.max_val = max_val
    
    def digitize_np(self, data):
        bins = np.linspace(self.min_val, self.max_val, self.vocab_size-1)
        return np.digitize(data, bins)
    def de_digitize_np(self, token):
        bin_width = (self.max_val - self.min_val) / self.vocab_size
        return self.min_val + token * bin_width + bin_width / 2  # return center of the bin
    
class WindowNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def normalize(self, window, params=None):
        window = np.asarray(window)  # Ensure it's a numpy array
        if not params:
            params = {'min': window.min(-1)[...,None], 'max': window.max(-1)[...,None]}
        normalized = (window - params['min']) / (params['max'] - params['min'] + self.epsilon)
        
        return normalized, params
    
    def denormalize(self, normalized_value, params):
        return normalized_value * (params['max'] - params['min'] + self.epsilon) + params['min']

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_window, vocab_size):
        self.data = data
        self.context_window = context_window
        self.vocab_size = vocab_size
        self.n_episodes, self.episode_length = data.shape
        self.samples_per_episode = self.episode_length - context_window
        self.normalizer = WindowNormalizer()
        self.digitizer = UniformDigitizer(vocab_size)
    def __len__(self):
        return self.n_episodes * self.samples_per_episode
    
    def __getitem__(self, idx):
        # Convert flat index to (episode, position)
        episode_idx = idx // self.samples_per_episode
        pos = idx % self.samples_per_episode
        
        x = self.data[episode_idx, pos:pos+self.context_window]
        y = self.data[episode_idx, pos+self.context_window]
        # normalize
        x_norm, par = self.normalizer.normalize(x)
        y_norm,_ = self.normalizer.normalize(y, par)

        # digitize
        x_dig = self.digitizer.digitize_np(x_norm)
        y_dig = self.digitizer.digitize_np(y_norm)


        return torch.tensor(x_dig, dtype=torch.long), torch.tensor(y_dig, dtype=torch.long).squeeze()
    

class TokenPositionEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, context_window, embedding_dim):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = torch.nn.Embedding(context_window, embedding_dim)
    def forward(self, x):
        # embedding will be added as the last dimention
        tex = self.token_embed(x)
        pex = self.pos_embed(torch.arange(x.size(-1), device=x.device).unsqueeze(0))
        return tex+pex



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        
        # Q, K, V projections
        self.qproject = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.kproject = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.vproject = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        # Output projection
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        self.out_project = torch.nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.layernorm = torch.nn.LayerNorm(embedding_dim)
        
    def forward(self,x):
        from math import sqrt
        #remember input is 32,40,128
        q=self.qproject(x)
        k=self.kproject(x)
        v=self.vproject(x)

        # reshape to heads
        q = torch.reshape(q, (q.size(0),-1,self.num_heads, self.head_dim)).permute(0,2,1,3) # 32,8,40,16
        k = torch.reshape(k, (k.size(0),-1,self.num_heads, self.head_dim)).permute(0,2,1,3) # 32,8,40,16
        v = torch.reshape(v, (v.size(0),-1,self.num_heads, self.head_dim)).permute(0,2,1,3) # 32,8,40,16

        score = q@k.permute(0,1,3,2) / sqrt(self.head_dim) # 32,8,40,40
        soft_score = torch.softmax(score, dim=-1) # 32,8,40,40
        scoreV = soft_score@v  # 32,8,40,16
        sv = scoreV.permute(0,2,1,3).contiguous().view(scoreV.size(0), scoreV.size(2), -1)        
        mha_out = self.out_project(sv) #32,40,128
        mha_out_plus_input = mha_out + x #32,40,128
        out = self.layernorm(mha_out_plus_input) #32,40,128
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.first_proj = torch.nn.Linear(embedding_dim, int(4*embedding_dim))
        self.relu = torch.nn.ReLU()
        self.second_proj = torch.nn.Linear(int(4*embedding_dim), embedding_dim)
        self.layernorm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x is shape 32, 40, 128
        x_proj1 = self.first_proj(x) # 32, 40, 4*128
        x_proj1_relu = self.relu(x_proj1)# 32, 40, 4*128
        x_proj2 = self.second_proj(x_proj1_relu)# 32, 40, 128
        x_proj2_plus_input = x_proj2 + x # 32, 40, 4*128
        out = self.layernorm(x_proj2_plus_input) # 32, 40, 4*128
        return out



class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        # Add your MultiHeadAttention and FeedForward here
        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x):
        mha_out = self.mha(x)
        out = self.ff(mha_out)
        return out





class DegradationTransformer(torch.nn.Module):
    def __init__(self, vocab_size, context_window, embedding_dim, num_heads, num_blocks):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.tpembed = TokenPositionEmbedding(vocab_size, context_window, embedding_dim)
        self.tbls_list = torch.nn.ModuleList([TransformerBlock(embedding_dim, num_heads) for _ in range(num_blocks)])
        self.lm_head = torch.nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tpembed.token_embed.weight  # Weight tying

    def forward(self, x):
        # input is 32*40
        x = self.tpembed(x) # 32*40*128
        for block in self.tbls_list:
            x = block(x) # 32*40*128
        last_x = x[:,-1, :] # 32*128
        vocab_x = self.lm_head(last_x) # 32*300
        return vocab_x

# export learner.py
class Learner():
    def __init__(self, model, optim, loss_func, train_loader, test_loader, cbs, device=None):
        # device: torch.device or string like 'cpu'/'cuda'. If None, auto-detect.
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # move model to device
        self.model = model.to(self.device)
        self.optim = optim
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cbs = cbs
        self.train_losses = []
        self.test_losses = []

        self.normalizer = WindowNormalizer()
        self.digitizer = UniformDigitizer(model.vocab_size)

    def train_one_batch(self, x_batch, y_batch):
        self.optim.zero_grad()
        # Move batch to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        # Forward pass
        y_predict = self.model(x_batch)
        # Compute loss
        loss = self.loss_func(y_predict, y_batch)
        # Backward pass
        loss.backward()
        # Update weights
        self.optim.step()
        return loss

    def test_one_batch(self, x_batch, y_batch):
        with torch.no_grad():        
            # Move batch to device
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            # Forward pass
            y_test_predict = self.model(x_batch)
            # Compute loss
            loss = self.loss_func(y_test_predict, y_batch)

            return loss

    def fit(self, num_epochs):
        for cb in self.cbs:
                cb.before_fit(self)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            for cb in self.cbs:
                cb.before_epoch(self)
            
            # train loop
            for i, (x_batch, y_batch) in enumerate(self.train_loader):
                self.train_idx = i
                for cb in self.cbs:
                    cb.before_train_batch(self)
                self.last_train_loss = self.train_one_batch(x_batch, y_batch)
                self.train_losses.append(self.last_train_loss)
                for cb in self.cbs:
                    cb.after_train_batch(self)
               
            # test loop
            for i, (x_batch, y_batch) in enumerate(self.test_loader):
                self.test_idx = i
                for cb in self.cbs:
                    cb.before_test_batch(self)
                self.last_test_loss = self.test_one_batch(x_batch, y_batch)
                self.test_losses.append(self.last_test_loss)
                for cb in self.cbs:
                    cb.after_test_batch(self)


            for cb in self.cbs:
                cb.after_epoch(self)

        for cb in self.cbs:
                cb.after_fit(self)

    def predict(self, x, num_periods=60):
        """
        Inference with the trained model (PyTorch version)
        x: numpy array, 1D or 2D
        """
        self.model.eval()
        
        # Convert to torch and add batch dim if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            for _ in range(num_periods):
                x_input = x[:, -self.model.context_window:]
                
                # Normalize (convert to numpy for normalizer)
                x_np = x_input.cpu().numpy()
                x_norm, par = self.normalizer.normalize(x_np)
                
                # Digitize and back to torch
                x_dig = self.digitizer.digitize_np(x_norm)
                x_dig = torch.tensor(x_dig, dtype=torch.long, device=self.device)
                # Inference
                y_out = self.model(x_dig)
                # Get predicted token
                yhat_token = torch.argmax(y_out, dim=-1).unsqueeze(-1)

                # De-digitize and denormalize
                yhat_norm = self.digitizer.de_digitize_np(yhat_token.cpu().numpy())
                predicted_y = self.normalizer.denormalize(yhat_norm, par)
                # Append to x (as torch tensor)
                predicted_y_torch = torch.tensor(predicted_y, dtype=torch.float32, device=self.device)
                x = torch.cat([x, predicted_y_torch], dim=1)
        
        return x.cpu().numpy()
    def save_model(self, filename="model.safetensors"):
        from safetensors.torch import save_model
        save_model(self.model, filename)

class Callback:
    def __init__(self): pass
    def before_fit(self, learner): pass
    def before_train_batch(self, learner): pass
    def before_test_batch(self, learner): pass
    def before_epoch(self, learner): pass
    def after_train_batch(self, learner): pass
    def after_test_batch(self, learner): pass
    def after_epoch(self, learner): pass
    def after_fit(self, learner): pass

class LogitDistributionCallback(Callback):
    def __init__(self, sample_freq=1):
        self.sample_freq = sample_freq  # How often to plot (every N epochs)
    
    def after_epoch(self, learner):
        if learner.epoch % self.sample_freq == 0:
            # Get a test batch
            x_batch, y_batch = next(iter(learner.test_loader))
            # Move batch to device and get model predictions (logits)
            x_batch = x_batch.to(learner.device)
            y_batch = y_batch.to(learner.device)
            with torch.no_grad():
                y_predict = learner.model(x_batch)
            # For a few samples, plot logits around the target
            probs = torch.softmax(y_predict, dim=-1)
            probs_cpu = probs.cpu().numpy()
            y_cpu = y_batch.cpu().numpy()
            for i in range(probs_cpu.shape[0]):
                plt.plot(range(probs_cpu.shape[1]), probs_cpu[i])
                plt.bar(int(y_cpu[i]), 1, color='red', width=2)
                plt.show()

class ProgressCallback(Callback):
    def __init__(self, update_freq=50):
        self.update_freq = update_freq
        self.train_losses = []
        self.test_losses = []

    def before_fit(self, learner):
        clear_output(wait=True)
        self.train_losses = []
        self.test_losses = []

    def after_train_batch(self, learner):
        if learner.train_idx % self.update_freq == 0:
            self.train_losses = [loss.item() for loss in learner.train_losses]
            self._update_plot()

    def after_test_batch(self, learner):
        if learner.test_idx % self.update_freq == 0:
            self.test_losses = [loss.item() for loss in learner.test_losses]
            self._update_plot()

    def _update_plot(self):
        clear_output(wait=True)
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Training Loss", "Test Loss"),
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        fig.add_trace(
            go.Scatter(x=np.arange(len(self.train_losses)), y=self.train_losses, mode='lines', name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=np.arange(len(self.test_losses)), y=self.test_losses, mode='lines', name='Test Loss', line=dict(color='orange')),
            row=2, col=1
        )
        fig.update_layout(height=600, width=800, showlegend=True, margin=dict(t=50, b=50, l=50, r=50))
        fig.update_xaxes(title_text="Batch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.show()

    def after_fit(self, learner):
        clear_output(wait=True)
        self.train_losses = []
        self.test_losses = []