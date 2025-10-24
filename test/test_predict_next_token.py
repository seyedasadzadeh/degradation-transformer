
from src.utils import WindowNormalizer
import numpy as np

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