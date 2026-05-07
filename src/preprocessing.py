import numpy as np
import torch

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
    def __init__(self, vocab_size, min_val=0, max_val=1.3):
        self.vocab_size = vocab_size
        self.min_val = min_val
        self.max_val = max_val
    
    def digitize_np(self, data):
        bins = np.linspace(self.min_val, self.max_val, self.vocab_size-1)
        return np.digitize(data, bins)

    def de_digitize_np(self, token):
        bin_width = (self.max_val - self.min_val) / self.vocab_size
        return self.min_val + token * bin_width + bin_width / 2  # return center of the bin

    def digitize_torch(self, data):
        # data is tensor
        device = data.device
        bins = torch.linspace(self.min_val, self.max_val, self.vocab_size-1, device=device)
        tokens = torch.bucketize(data, bins)
        return torch.clamp(tokens, 0, self.vocab_size-1)

    def de_digitize_torch(self, token):
        # token is tensor
        bin_width = (self.max_val - self.min_val) / self.vocab_size
        return self.min_val + token.float() * bin_width + bin_width / 2
    
class WindowNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def normalize(self, window, params=None):
        if isinstance(window, torch.Tensor):
            if params is None:
                min_val = window.min(dim=-1, keepdim=True)[0]
                max_val = window.max(dim=-1, keepdim=True)[0]
                params = {'min': min_val, 'max': max_val}
            normalized = (window - params['min']) / (params['max'] - params['min'] + self.epsilon)
            return normalized, params
        else:
            window = np.asarray(window)  # Ensure it's a numpy array
            if not params:
                params = {'min': window.min(-1)[...,None], 'max': window.max(-1)[...,None]}
            normalized = (window - params['min']) / (params['max'] - params['min'] + self.epsilon)
            return normalized, params
    
    def denormalize(self, normalized_value, params):
        return normalized_value * (params['max'] - params['min'] + self.epsilon) + params['min']


def context_metadata(window, epsilon=1e-8):
    """
    Compute scale/location/rate features for a raw context window.

    Features are: min, max, range, last value, mean slope, slope std.
    Supports NumPy arrays and torch tensors with shape (context,) or
    (batch, context).
    """
    if isinstance(window, torch.Tensor):
        if window.ndim == 1:
            window = window.unsqueeze(0)
        min_val = window.min(dim=-1).values
        max_val = window.max(dim=-1).values
        value_range = max_val - min_val
        last_value = window[..., -1]
        if window.shape[-1] > 1:
            slopes = torch.diff(window, dim=-1)
            mean_slope = slopes.mean(dim=-1)
            std_slope = slopes.std(dim=-1, unbiased=False)
        else:
            mean_slope = torch.zeros_like(last_value)
            std_slope = torch.zeros_like(last_value)
        features = torch.stack([min_val, max_val, value_range, last_value, mean_slope, std_slope], dim=-1)
        return torch.nan_to_num(features, nan=0.0, posinf=1.0 / epsilon, neginf=-1.0 / epsilon)

    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 1:
        window = window[None, :]
    min_val = window.min(axis=-1)
    max_val = window.max(axis=-1)
    value_range = max_val - min_val
    last_value = window[..., -1]
    if window.shape[-1] > 1:
        slopes = np.diff(window, axis=-1)
        mean_slope = slopes.mean(axis=-1)
        std_slope = slopes.std(axis=-1)
    else:
        mean_slope = np.zeros_like(last_value)
        std_slope = np.zeros_like(last_value)
    features = np.stack([min_val, max_val, value_range, last_value, mean_slope, std_slope], axis=-1)
    return np.nan_to_num(features, nan=0.0, posinf=1.0 / epsilon, neginf=-1.0 / epsilon).astype(np.float32)


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_window, vocab_size, return_metadata=True):
        self.data = data
        self.context_window = context_window
        self.vocab_size = vocab_size
        self.return_metadata = return_metadata
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

        x_tensor = torch.tensor(x_dig, dtype=torch.long)
        y_tensor = torch.tensor(y_dig, dtype=torch.long).squeeze()
        if not self.return_metadata:
            return x_tensor, y_tensor
        metadata_tensor = torch.tensor(context_metadata(x).squeeze(0), dtype=torch.float32)
        return x_tensor, metadata_tensor, y_tensor
