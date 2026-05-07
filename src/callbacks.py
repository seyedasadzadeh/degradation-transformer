import numpy as np
import torch
import matplotlib.pyplot as plt

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
            batch = next(iter(learner.test_loader))
            x_batch, metadata_batch, y_batch = learner._unpack_supervised_batch(batch)
            # Move batch to device and get model predictions (logits)
            x_batch = x_batch.to(learner.device)
            if metadata_batch is not None:
                metadata_batch = metadata_batch.to(learner.device)
            y_batch = y_batch.to(learner.device)
            with torch.no_grad():
                y_predict = learner.model(x_batch, metadata_batch)
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
        from IPython.display import clear_output
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
        from IPython.display import clear_output
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
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
        from IPython.display import clear_output
        clear_output(wait=True)
        self.train_losses = []
        self.test_losses = []

class SaveModel(Callback):
    def __init__(self, file_name="degradation_transformer_model.safetensors"):
        self.file_name = file_name
    def after_fit(self, learner):
        # save model files
        learner.save_model(self.file_name) # this saves config as well

class WandBCallback(Callback):
    def __init__(self, update_freq=50):
        self.update_freq = update_freq
        self.train_losses = []
        self.test_losses = []

    def before_fit(self, learner):
        import wandb
        self.run = wandb.init(project="degradation-transformer",
                   config = {
                       'vocab_size': learner.model.vocab_size,
                        'context_window': learner.model.context_window,
                        'embedding_dim': learner.model.tpembed.token_embed.embedding_dim,
                        'num_heads': learner.model.tbls_list[0].mha.num_heads,
                        'num_blocks': len(learner.model.tbls_list),
                        "learning_rate": learner.optim.param_groups[0]['lr'],
                        "epochs": learner.num_epochs
                        }
                    )
        self.run.log({"train_size": len(learner.train_loader), "test_size": len(learner.test_loader)})


    def after_train_batch(self, learner):
        if learner.train_idx % self.update_freq == 0:
            self.train_losses = [loss.item() for loss in learner.train_losses]
            self.run.log({"train_loss": self.train_losses[-1]})

    def after_test_batch(self, learner):
        if learner.test_idx % self.update_freq == 0:
            self.test_losses = [loss.item() for loss in learner.test_losses]
            self.run.log({"test_loss": self.test_losses[-1]})
            

    def after_epoch(self, learner):
        self.run.log({"last_test_loss": sum(self.test_losses[-50:]) / len(self.test_losses[-50:]), 
                      "epoch": learner.epoch})
    
    def after_fit(self, learner):
       

        # create artifact
        import wandb
        artifact = wandb.Artifact('degradation-transformer-model', type='model')
        artifact.add_file('degradation_transformer_model.safetensors')
        artifact.add_file('degradation_transformer_model_config.json')
        self.run.log_artifact(artifact)

        self.run.finish()






class MLflowCallback(Callback):
    def __init__(self, update_freq=50):
        self.update_freq = update_freq
        self.train_losses = []
        self.test_losses = []

    def before_fit(self, learner):
        import mlflow
        mlflow.start_run()
        mlflow.log_params({
                       'vocab_size': learner.model.vocab_size,
                        'context_window': learner.model.context_window,
                        'embedding_dim': learner.model.tpembed.token_embed.embedding_dim,
                        'num_heads': learner.model.tbls_list[0].mha.num_heads,
                        'num_blocks': len(learner.model.tbls_list),
                        "learning_rate": learner.optim.param_groups[0]['lr'],
                        "epochs": learner.num_epochs
                        }
                    )
        mlflow.log_metric("train_size", len(learner.train_loader))
        mlflow.log_metric("test_size", len(learner.test_loader))


    def after_train_batch(self, learner):
        import mlflow
        if learner.train_idx % self.update_freq == 0:
            self.train_losses = [loss.item() for loss in learner.train_losses]
            mlflow.log_metric("train_loss", self.train_losses[-1], step=learner.train_idx)

    def after_test_batch(self, learner):
        import mlflow
        if learner.test_idx % self.update_freq == 0:
            self.test_losses = [loss.item() for loss in learner.test_losses]
            mlflow.log_metric("test_loss", self.test_losses[-1], step=learner.test_idx)
            

    def after_epoch(self, learner):
        import mlflow
        mlflow.log_metric("last_test_loss", sum(self.test_losses[-50:]) / len(self.test_losses[-50:]), 
                      step=learner.epoch)
    
    def after_fit(self, learner):
        import mlflow
        import mlflow.pytorch
        import torch

        # Put model in eval mode
        learner.model.eval()
        
        # Get device from model
        device = next(learner.model.parameters()).device
        
        # Create example token input.
        example_input = torch.randint(
            0,
            learner.model.vocab_size,
            (1, learner.model.context_window),
            device=device
        )
        
        # Generate example output (forward pass, no grad)
        with torch.no_grad():
            example_output = learner.model(example_input)
        
        # Convert to numpy for MLflow
        example_input_np = example_input.cpu().numpy()
        example_output_np = example_output.cpu().numpy()

        # Start MLflow run (if not already active)
        if not mlflow.active_run():
            mlflow.start_run()

        # Log model with both input and output examples
        mlflow.pytorch.log_model(
            pytorch_model=learner.model,
            artifact_path="model",
            input_example=example_input_np,
            #output_example=example_output_np  # This is critical!
        )

        mlflow.end_run()
