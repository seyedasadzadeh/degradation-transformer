import numpy as np
import torch

from .preprocessing import UniformDigitizer, WindowNormalizer, context_metadata

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

    def train_one_batch(self, x_batch, y_batch, metadata_batch=None, attention_mask=None):
        self.optim.zero_grad()
        # Move batch to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        if metadata_batch is not None:
            metadata_batch = metadata_batch.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # Forward pass
        y_predict = self.model(x_batch, metadata_batch, attention_mask=attention_mask)
        # Compute loss
        loss = self.loss_func(y_predict, y_batch)
        # Backward pass
        loss.backward()
        # Update weights
        self.optim.step()
        return loss

    def test_one_batch(self, x_batch, y_batch, metadata_batch=None, attention_mask=None):
        with torch.no_grad():        
            # Move batch to device
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            if metadata_batch is not None:
                metadata_batch = metadata_batch.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            # Forward pass
            y_test_predict = self.model(x_batch, metadata_batch, attention_mask=attention_mask)
            # Compute loss
            loss = self.loss_func(y_test_predict, y_batch)

            return loss

    def _unpack_supervised_batch(self, batch):
        if len(batch) == 2:
            x_batch, y_batch = batch
            metadata_batch = None
            attention_mask = None
        elif len(batch) == 3:
            x_batch, metadata_batch, y_batch = batch
            attention_mask = None
        elif len(batch) == 4:
            x_batch, metadata_batch, attention_mask, y_batch = batch
        else:
            raise ValueError(f"Expected supervised batch with 2, 3, or 4 tensors, got {len(batch)}.")
        return x_batch, metadata_batch, attention_mask, y_batch

    def fit(self, num_epochs):
        self.num_epochs = num_epochs
        for cb in self.cbs:
                cb.before_fit(self)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            for cb in self.cbs:
                cb.before_epoch(self)
            
            # train loop
            for i, batch in enumerate(self.train_loader):
                x_batch, metadata_batch, attention_mask, y_batch = self._unpack_supervised_batch(batch)
                self.train_idx = i
                for cb in self.cbs:
                    cb.before_train_batch(self)
                self.last_train_loss = self.train_one_batch(x_batch, y_batch, metadata_batch, attention_mask)
                self.train_losses.append(self.last_train_loss)
                for cb in self.cbs:
                    cb.after_train_batch(self)
               
            # test loop
            for i, batch in enumerate(self.test_loader):
                x_batch, metadata_batch, attention_mask, y_batch = self._unpack_supervised_batch(batch)
                self.test_idx = i
                for cb in self.cbs:
                    cb.before_test_batch(self)
                self.last_test_loss = self.test_one_batch(x_batch, y_batch, metadata_batch, attention_mask)
                self.test_losses.append(self.last_test_loss)
                for cb in self.cbs:
                    cb.after_test_batch(self)


            for cb in self.cbs:
                cb.after_epoch(self)

        for cb in self.cbs:
                cb.after_fit(self)

    def predict(self, x, num_periods=60, temperature=1.0, top_k=None, top_p=None, grad=False, return_tensor=False):
        """
        Inference with the trained model (PyTorch version)
        
        Args:
            x: numpy array, 1D or 2D - initial context
            num_periods: int - number of future steps to predict
            temperature: float - sampling temperature
                - 0.0: greedy (argmax)
                - 1.0: sample from model distribution
                - >1.0: more random (flatter distribution)
            top_k: int or None - if set, only sample from top k tokens
            top_p: float or None - if set, nucleus sampling (sample from smallest set with cumulative prob > p)
        """
        self.model.eval()
        
        # Convert to torch and add batch dim if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        input_length = x.shape[-1]
        min_context_window = getattr(self.model, "min_context_window", None)
        if min_context_window is not None and input_length < min_context_window:
            raise ValueError(
                f"Input context length {input_length} is shorter than the model minimum "
                f"context length {min_context_window}."
            )
        
        import contextlib
        log_probs = []
        yhat_tokens = torch.empty((x.size(0), 0), dtype=torch.long, device=self.device)

        ctx = torch.no_grad() if not grad else contextlib.nullcontext()
        with ctx:
            for _ in range(num_periods):
                x_input = x[:, -self.model.context_window:]
                real_context_len = x_input.shape[-1]
                
                # Normalize (convert to numpy for normalizer)
                x_np = x_input.cpu().numpy()
                x_norm, par = self.normalizer.normalize(x_np)
                metadata = None
                if getattr(self.model, "metadata_dim", 0) > 0:
                    metadata_kwargs = {}
                    if getattr(self.model, "use_padding", False):
                        metadata_kwargs = {
                            "context_length": real_context_len,
                            "max_context_window": self.model.context_window,
                        }
                    metadata = context_metadata(x_np, **metadata_kwargs)
                    metadata = torch.tensor(metadata, dtype=torch.float32, device=self.device)
                
                # Digitize and back to torch
                x_dig = self.digitizer.digitize_np(x_norm)
                attention_mask = None
                if getattr(self.model, "use_padding", False) and real_context_len < self.model.context_window:
                    padded = np.full(
                        (x_dig.shape[0], self.model.context_window),
                        self.model.pad_token_id,
                        dtype=np.int64,
                    )
                    padded[:, -real_context_len:] = x_dig
                    attention_mask_np = np.zeros((x_dig.shape[0], self.model.context_window), dtype=np.float32)
                    attention_mask_np[:, -real_context_len:] = 1.0
                    x_dig = padded
                    attention_mask = torch.tensor(attention_mask_np, dtype=torch.float32, device=self.device)
                elif getattr(self.model, "use_padding", False):
                    attention_mask = torch.ones(x_dig.shape, dtype=torch.float32, device=self.device)
                x_dig = torch.tensor(x_dig, dtype=torch.long, device=self.device)
                
                # Inference
                logits = self.model(x_dig, metadata, attention_mask=attention_mask)  # (batch_size, vocab_size)
                
                # Apply temperature
                if temperature == 0:
                    # Greedy decoding
                    yhat_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                    # If gradients are requested (RL), compute the log-prob of the selected token
                    if grad:
                        logp = torch.log_softmax(logits, dim=-1).gather(1, yhat_token).squeeze(-1)
                        log_probs.append(logp)
                else:
                    # Temperature scaling
                    logits = logits / temperature
                    
                    # Top-k filtering
                    if top_k is not None:
                        top_k = min(top_k, logits.size(-1))
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = -float('Inf')
                    
                    # Top-p (nucleus) filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -float('Inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()       # token index
                    # log_prob should be connected to the model parameters when grad=True
                    lp = dist.log_prob(action)
                    # always store lp (it will be a tensor; if grad=False it's detached by context)
                    log_probs.append(lp)
                    yhat_token = action.unsqueeze(-1)
 
                # De-digitize and denormalize
                yhat_norm = self.digitizer.de_digitize_np(yhat_token.cpu().numpy())
                predicted_y = self.normalizer.denormalize(yhat_norm, par)
                # Append to x (as torch tensor)
                predicted_y_torch = torch.tensor(predicted_y, dtype=torch.float32, device=self.device)
                x = torch.cat([x, predicted_y_torch], dim=1)
                yhat_tokens = torch.cat([yhat_tokens, yhat_token], dim=1)
        preds_tensor = x[:, input_length:]

        # Enforce API contract: grad requires tensor outputs to keep autograd
        if grad and not return_tensor:
            raise ValueError("predict(..., grad=True) requires return_tensor=True so log_probs remain connected to autograd.")

        if return_tensor:
            return preds_tensor, log_probs, yhat_tokens
        else:
            preds_np = preds_tensor.cpu().numpy()
            if log_probs:
                logp_np = [lp.detach().cpu().numpy() for lp in log_probs]
            else:
                logp_np = None
            return preds_np, logp_np

    def forecast_metrics(self, data, num_periods=60, **kwargs):
        from .evaluation import forecast_metrics

        return forecast_metrics(self, data, num_periods=num_periods, **kwargs)
    
    def save_model(self, filename="model.safetensors"):
        from safetensors.torch import save_model
        import json
        
        # Save model weights
        save_model(self.model, filename)
        
        # Save config
        config = {
            'vocab_size': self.model.vocab_size,
            'context_window': self.model.context_window,
            'embedding_dim': self.model.tpembed.token_embed.embedding_dim,
            'num_heads': self.model.tbls_list[0].mha.num_heads,
            'num_blocks': len(self.model.tbls_list),
            'metadata_dim': getattr(self.model, 'metadata_dim', 0),
            'pad_token_id': getattr(self.model, 'pad_token_id', None),
            'use_padding': getattr(self.model, 'use_padding', False),
            'min_context_window': getattr(self.model, 'min_context_window', None),
        }
        config_filename = filename.replace('.safetensors', '_config.json')
        with open(config_filename, 'w') as f:
            json.dump(config, f)
