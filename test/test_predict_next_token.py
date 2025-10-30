
import numpy as np
import json

from src.utils import WindowNormalizer
from src.utils import DegradationTransformer
from src.utils import Learner

from safetensors.torch import load_model


def test_normalizer_2d():
    normalizer = WindowNormalizer()
    window = np.array([[1, 2, 3], [4, 5, 6]])
    normalized, params = normalizer.normalize(window)
    assert np.allclose(normalized[...,-1], 1), "normalization is not correct"
    assert np.allclose(normalized[...,0], 0), "normalization is not correct"
    denormalized = normalizer.denormalize(normalized, params)
    assert np.allclose(window, denormalized), "Denormalization did not return the original window"
    
    
def test_normalizer_1d():
    normalizer = WindowNormalizer()
    window = np.array([1, 2, 3])
    normalized, params = normalizer.normalize(window)
    assert np.allclose(normalized[...,-1], 1), "normalization is not correct"
    assert np.allclose(normalized[...,0], 0), "normalization is not correct"
    denormalized = normalizer.denormalize(normalized, params)
    assert np.allclose(window, denormalized), "Denormalization did not return the original window"
    denormalized_0 = normalizer.denormalize(normalized[0], params)
    assert np.allclose(window[0], denormalized_0), "Denormalization did not return the original element"


def test_load_model_from_safetensors_locally():
    
    x_batch = np.random.randint(0, 100, size=(2, 10))

    with open("degradation_transformer_model_config.json", "rb") as f:
        model_params=json.load(f)

    model = DegradationTransformer(vocab_size=model_params['vocab_size'], 
                                    context_window=model_params['context_window'], 
                                embedding_dim=model_params['embedding_dim'], 
                                num_heads=model_params['num_heads'],
                                    num_blocks=model_params['num_blocks'])
    load_model(model, "degradation_transformer_model.safetensors")
    model.eval()
    learner = Learner(model, optim=None, loss_func=None, 
                    train_loader=None, test_loader=None, cbs=[])

    y_predict = learner.predict(x_batch, num_periods=10)
    assert y_predict.shape == (2, 20), "Prediction shape is not correct"
