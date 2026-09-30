import numpy as np
import torch

from src.experiments.deep import CNN1DClassifier, GRUClassifier, LSTMClassifier, MLPClassifier, parameter_count


def test_mlp_and_cnn_forward_shapes_and_parameter_counts():
    mlp = MLPClassifier(36, 4)
    cnn = CNN1DClassifier(3, 4)
    gru = GRUClassifier(3, 4)
    lstm = LSTMClassifier(3, 4)
    assert mlp(torch.zeros(2, 36)).shape == (2, 4)
    assert cnn(torch.zeros(2, 3, 20)).shape == (2, 4)
    assert gru(torch.zeros(2, 3, 20)).shape == (2, 4)
    assert lstm(torch.zeros(2, 3, 20)).shape == (2, 4)
    assert parameter_count(mlp) > 0
    assert parameter_count(cnn) > 0