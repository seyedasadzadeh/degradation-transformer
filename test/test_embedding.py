
import torch
import json
from safetensors.torch import load_model
from src.utils import DegradationTransformer
def test_embedding():
    # Load Config
    with open('degradation_transformer_model_config.json', 'r') as f:
        config = json.load(f)

    # Initialize Model
    model = DegradationTransformer(
    vocab_size=config['vocab_size'],
    context_window=config['context_window'],
    embedding_dim=config['embedding_dim'],
    num_heads=config['num_heads'],
    num_blocks=config['num_blocks']
)

    # Load Weights
    load_model(model, 'degradation_transformer_model.safetensors')
    print(f"Pos embed shape: {model.tpembed.pos_embed.weight.shape}")

    # Create dummy input
    x = torch.randint(0, config['vocab_size'], (32, 40))
    print(f"Input shape: {x.shape}")

    # Run forward
    out = model.tpembed.token_embed(x)
    print("Forward pass successful!")
    assert out.shape == (32, 40, config['embedding_dim']) # (batch_size, seq_len, embedding_dim)