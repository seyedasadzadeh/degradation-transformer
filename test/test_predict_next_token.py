
import numpy as np
import json

from src.evaluation import forecast_metrics
from src.learner import Learner
from src.model import DegradationTransformer
from src.preprocessing import TimeSeriesDataset
from src.preprocessing import WindowNormalizer
from src.preprocessing import context_metadata

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
    
    with open("degradation_transformer_model_config.json", "rb") as f:
        model_params=json.load(f)

    x_batch = np.random.randint(0, 100, size=(2, model_params["context_window"]))

    model = DegradationTransformer(vocab_size=model_params['vocab_size'], 
                                    context_window=model_params['context_window'], 
                                embedding_dim=model_params['embedding_dim'], 
                                num_heads=model_params['num_heads'],
                                    num_blocks=model_params['num_blocks'],
                                    metadata_dim=model_params.get('metadata_dim', 0))
    load_model(model, "degradation_transformer_model.safetensors")
    model.eval()
    learner = Learner(model, optim=None, loss_func=None, 
                    train_loader=None, test_loader=None, cbs=[])

    y_predict = learner.predict(x_batch, num_periods=30)
    assert y_predict[0].shape == (2, 30), "Prediction shape is not correct"


def test_context_metadata_values():
    window = np.array([[1, 2, 4, 7], [3, 3, 3, 3]], dtype=np.float32)
    metadata = context_metadata(window)
    expected_first = np.array([1, 7, 6, 7, 2, np.std([1, 2, 3])], dtype=np.float32)
    expected_second = np.array([3, 3, 0, 3, 0, 0], dtype=np.float32)
    assert metadata.shape == (2, 6)
    assert np.allclose(metadata[0], expected_first)
    assert np.allclose(metadata[1], expected_second)


def test_dataset_returns_metadata_and_conditioned_model_predicts():
    data = np.stack([
        np.linspace(0, 1, 20),
        np.linspace(2, 5, 20),
    ]).astype(np.float32)
    dataset = TimeSeriesDataset(data, context_window=8, vocab_size=32)
    x_tokens, metadata, y_token = dataset[0]
    assert x_tokens.shape == (8,)
    assert metadata.shape == (6,)
    assert y_token.shape == ()

    model = DegradationTransformer(
        vocab_size=32,
        context_window=8,
        embedding_dim=16,
        num_heads=4,
        num_blocks=1,
        metadata_dim=6,
    )
    learner = Learner(model, optim=None, loss_func=None, train_loader=None, test_loader=None, cbs=[], device="cpu")
    y_predict, _ = learner.predict(data[:, :8], num_periods=3, temperature=0.0)
    assert y_predict.shape == (2, 3)


def test_predict_returns_only_generated_steps_for_longer_context():
    model = DegradationTransformer(
        vocab_size=32,
        context_window=4,
        embedding_dim=16,
        num_heads=4,
        num_blocks=1,
        metadata_dim=6,
    )
    learner = Learner(model, optim=None, loss_func=None, train_loader=None, test_loader=None, cbs=[], device="cpu")
    x = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float32)

    y_predict, _ = learner.predict(x, num_periods=3, temperature=0.0)

    assert y_predict.shape == (1, 3)


def test_forecast_metrics_uses_sliding_windows_and_reports_errors():
    class ZeroLearner:
        class Model:
            context_window = 2
        model = Model()

        def predict(self, x, num_periods=1, temperature=0.0):
            return np.zeros((x.shape[0], num_periods), dtype=np.float32), None

    data = np.array([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ], dtype=np.float32)

    metrics = forecast_metrics(
        ZeroLearner(),
        data,
        num_periods=2,
        batch_size=2,
        return_predictions=True,
    )

    y_true = np.array([
        [2, 3],
        [3, 4],
        [3, 4],
        [4, 5],
    ], dtype=np.float32)
    assert metrics["n_samples"] == 4
    assert metrics["x_context"].shape == (4, 2)
    assert metrics["y_true"].shape == (4, 2)
    assert metrics["y_pred"].shape == (4, 2)
    assert np.allclose(metrics["y_true"], y_true)
    assert np.isclose(metrics["mae"], np.mean(np.abs(y_true)))
    assert np.isclose(metrics["mse"], np.mean(y_true ** 2))
    assert np.isclose(metrics["final_mae"], np.mean(np.abs(y_true[:, -1])))
    assert metrics["per_horizon_mae"].shape == (2,)
    assert metrics["per_horizon_mse"].shape == (2,)
