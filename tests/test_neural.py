"""
Tests for neural network domain.
"""

import pytest
import numpy as np
from morphogen.stdlib.neural import (
    neural, DenseLayer, MLP, NeuralOperations
)


class TestActivationFunctions:
    """Test activation functions"""

    def test_tanh(self):
        """Test tanh activation"""
        x = np.array([0.0, 1.0, -1.0, 2.0])
        y = neural.tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(y, expected)

    def test_relu(self):
        """Test ReLU activation"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = neural.relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(y, expected)

    def test_sigmoid(self):
        """Test sigmoid activation"""
        x = np.array([0.0, 1.0, -1.0])
        y = neural.sigmoid(x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_array_almost_equal(y, expected)

    def test_sigmoid_numerical_stability(self):
        """Test sigmoid with large values"""
        x = np.array([-1000.0, 1000.0])
        y = neural.sigmoid(x)
        # Should not overflow/underflow
        assert np.all(np.isfinite(y))
        assert y[0] < 0.01  # Very negative → ~0
        assert y[1] > 0.99  # Very positive → ~1

    def test_softmax(self):
        """Test softmax activation"""
        x = np.array([1.0, 2.0, 3.0])
        y = neural.softmax(x)
        # Should sum to 1
        assert np.abs(np.sum(y) - 1.0) < 1e-6
        # Should be positive
        assert np.all(y > 0)

    def test_leaky_relu(self):
        """Test leaky ReLU activation"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = neural.leaky_relu(x, alpha=0.01)
        expected = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(y, expected)

    def test_apply_activation(self):
        """Test activation dispatcher"""
        x = np.array([1.0, 2.0, 3.0])

        y_linear = neural.apply_activation(x, 'linear')
        np.testing.assert_array_equal(y_linear, x)

        y_tanh = neural.apply_activation(x, 'tanh')
        np.testing.assert_array_almost_equal(y_tanh, np.tanh(x))

        y_relu = neural.apply_activation(x, 'relu')
        expected_relu = np.maximum(0, x)
        np.testing.assert_array_almost_equal(y_relu, expected_relu)


class TestLinearOperations:
    """Test linear transformations"""

    def test_linear_single(self):
        """Test linear transformation on single input"""
        x = np.array([1.0, 2.0, 3.0])
        weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        biases = np.array([0.5, 0.5])

        y = neural.linear(x, weights, biases)

        expected = np.array([1.5, 2.5])  # [1*1 + 0.5, 2*1 + 0.5]
        np.testing.assert_array_almost_equal(y, expected)

    def test_linear_batch(self):
        """Test linear transformation on batch"""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # Batch of 2
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        biases = np.array([0.5, 0.5])

        y = neural.linear(x, weights, biases)

        assert y.shape == (2, 2)
        # First sample: [1*1 + 2*0 + 0.5, 1*0 + 2*1 + 0.5] = [1.5, 2.5]
        np.testing.assert_array_almost_equal(y[0], [1.5, 2.5])


class TestDenseLayer:
    """Test dense layer operations"""

    def test_alloc_layer_xavier(self):
        """Test layer allocation with Xavier init"""
        layer = neural.alloc_layer(4, 8, activation='tanh', init_method='xavier', seed=42)

        assert layer.weights.shape == (4, 8)
        assert layer.biases.shape == (8,)
        assert layer.activation == 'tanh'
        # Xavier limit for [4, 8]: sqrt(6 / (4 + 8)) = sqrt(0.5) ≈ 0.707
        assert np.max(np.abs(layer.weights)) < 1.0

    def test_alloc_layer_deterministic(self):
        """Test deterministic layer initialization"""
        layer1 = neural.alloc_layer(4, 8, seed=42)
        layer2 = neural.alloc_layer(4, 8, seed=42)

        np.testing.assert_array_equal(layer1.weights, layer2.weights)
        np.testing.assert_array_equal(layer1.biases, layer2.biases)

    def test_dense_forward(self):
        """Test dense layer forward pass"""
        layer = neural.alloc_layer(3, 2, activation='linear', init_method='zeros')
        layer.weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        layer.biases = np.array([0.5, 0.5])

        x = np.array([1.0, 2.0, 3.0])
        y = neural.dense(x, layer)

        expected = np.array([1.5, 2.5])
        np.testing.assert_array_almost_equal(y, expected)

    def test_layer_copy(self):
        """Test layer copy semantics"""
        layer1 = neural.alloc_layer(4, 8, seed=42)
        layer2 = layer1.copy()

        np.testing.assert_array_equal(layer1.weights, layer2.weights)

        # Modify copy
        layer2.weights[0, 0] = 999.0
        assert layer1.weights[0, 0] != 999.0


class TestMLP:
    """Test multi-layer perceptron"""

    def test_alloc_mlp(self):
        """Test MLP allocation"""
        network = neural.alloc_mlp(
            layer_sizes=[4, 8, 1],
            activations=['tanh', 'sigmoid'],
            seed=42
        )

        assert network.input_size == 4
        assert network.output_size == 1
        assert len(network.layers) == 2
        assert network.layers[0].weights.shape == (4, 8)
        assert network.layers[1].weights.shape == (8, 1)

    def test_alloc_mlp_default_activations(self):
        """Test MLP with default activations"""
        network = neural.alloc_mlp([4, 8, 8, 1], seed=42)

        assert network.layers[0].activation == 'tanh'  # Hidden
        assert network.layers[1].activation == 'tanh'  # Hidden
        assert network.layers[2].activation == 'linear'  # Output

    def test_mlp_forward(self):
        """Test MLP forward pass"""
        network = neural.alloc_mlp([2, 3, 1], activations=['linear', 'linear'], seed=42)

        x = np.array([1.0, 2.0])
        y = neural.forward(x, network)

        assert y.shape == (1,)

    def test_mlp_forward_batch(self):
        """Test MLP forward pass with batch"""
        network = neural.alloc_mlp([2, 3, 1], activations=['linear', 'linear'], seed=42)

        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Batch of 3
        y = neural.forward(x, network)

        assert y.shape == (3, 1)

    def test_mlp_copy(self):
        """Test MLP copy semantics"""
        network1 = neural.alloc_mlp([4, 8, 1], seed=42)
        network2 = network1.copy()

        np.testing.assert_array_equal(
            network1.layers[0].weights,
            network2.layers[0].weights
        )

        # Modify copy
        network2.layers[0].weights[0, 0] = 999.0
        assert network1.layers[0].weights[0, 0] != 999.0

    def test_count_parameters(self):
        """Test parameter counting"""
        network = neural.alloc_mlp([4, 8, 1], seed=42)
        # [4,8]: 4*8 weights + 8 biases = 40
        # [8,1]: 8*1 weights + 1 bias = 9
        # Total: 49
        assert network.count_parameters() == 49


class TestParameterOperations:
    """Test parameter extraction and manipulation"""

    def test_get_parameters(self):
        """Test parameter extraction"""
        network = neural.alloc_mlp([2, 3, 1], seed=42)
        params = neural.get_parameters(network)

        # [2,3]: 2*3 + 3 = 9
        # [3,1]: 3*1 + 1 = 4
        # Total: 13
        assert params.shape == (13,)

    def test_set_parameters(self):
        """Test parameter setting"""
        network = neural.alloc_mlp([2, 3, 1], seed=42)
        params = neural.get_parameters(network)

        # Modify parameters
        new_params = params * 2.0
        new_network = neural.set_parameters(network, new_params)

        # Check parameters were set
        new_params_extracted = neural.get_parameters(new_network)
        np.testing.assert_array_almost_equal(new_params_extracted, new_params)

        # Original network should be unchanged
        old_params = neural.get_parameters(network)
        np.testing.assert_array_almost_equal(old_params, params)

    def test_mutate_parameters(self):
        """Test parameter mutation"""
        params = np.ones(10)
        mutated = neural.mutate_parameters(
            params,
            mutation_rate=1.0,  # Mutate all
            mutation_scale=0.1,
            seed=42
        )

        # Should be different from original
        assert not np.allclose(mutated, params)
        # But should be close (small mutation scale)
        assert np.allclose(mutated, params, atol=1.0)

    def test_mutate_parameters_rate(self):
        """Test mutation rate control"""
        params = np.ones(100)
        mutated = neural.mutate_parameters(
            params,
            mutation_rate=0.1,  # 10% mutation
            mutation_scale=10.0,
            seed=42
        )

        # Count how many changed significantly
        changed = np.sum(np.abs(mutated - params) > 0.01)
        # Should be around 10 ± 5
        assert 5 <= changed <= 20

    def test_crossover_uniform(self):
        """Test uniform crossover"""
        params1 = np.ones(10)
        params2 = np.zeros(10)

        child1, child2 = neural.crossover_parameters(
            params1, params2, method='uniform', seed=42
        )

        # Children should be mix of 0s and 1s
        assert np.all((child1 == 0) | (child1 == 1))
        assert np.all((child2 == 0) | (child2 == 1))
        # Should be complementary
        np.testing.assert_array_almost_equal(child1 + child2, np.ones(10))

    def test_crossover_single_point(self):
        """Test single-point crossover"""
        params1 = np.ones(10)
        params2 = np.zeros(10)

        child1, child2 = neural.crossover_parameters(
            params1, params2, method='single_point', seed=42
        )

        # Each child should have a run of 0s and 1s
        assert np.all((child1 == 0) | (child1 == 1))

    def test_crossover_blend(self):
        """Test blend crossover"""
        params1 = np.ones(10) * 2.0
        params2 = np.ones(10) * 4.0

        child1, child2 = neural.crossover_parameters(
            params1, params2, method='blend'
        )

        # Default alpha=0.5, so children should be 3.0
        np.testing.assert_array_almost_equal(child1, np.ones(10) * 3.0)
        np.testing.assert_array_almost_equal(child2, np.ones(10) * 3.0)


class TestFlappyBirdController:
    """Test Flappy Bird controller preset"""

    def test_flappy_bird_controller(self):
        """Test Flappy Bird controller creation"""
        controller = neural.flappy_bird_controller(seed=42)

        assert controller.input_size == 4
        assert controller.output_size == 1
        assert len(controller.layers) == 2
        assert controller.layers[0].activation == 'tanh'
        assert controller.layers[1].activation == 'sigmoid'
        assert controller.count_parameters() == 49


class TestIntegration:
    """Integration tests"""

    def test_train_xor(self):
        """Test network can learn simple XOR function"""
        # Create network
        network = neural.alloc_mlp([2, 4, 1], activations=['tanh', 'sigmoid'], seed=42)

        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)

        # Forward pass (just test it runs, not that it learns)
        outputs = neural.forward(X, network)

        assert outputs.shape == (4, 1)
        assert np.all(outputs >= 0) and np.all(outputs <= 1)

    def test_gradient_free_optimization(self):
        """Test that network parameters can be optimized via random search"""
        # Simple task: output constant 0.7
        target = 0.7

        def fitness(params):
            network_template = neural.alloc_mlp([1, 2, 1], seed=42)
            network = neural.set_parameters(network_template, params)
            output = neural.forward(np.array([[0.5]]), network)
            return -abs(output[0, 0] - target)  # Minimize error

        # Random search
        best_params = None
        best_fitness = -np.inf

        for i in range(100):
            network = neural.alloc_mlp([1, 2, 1], seed=i)
            params = neural.get_parameters(network)
            fit = fitness(params)
            if fit > best_fitness:
                best_fitness = fit
                best_params = params

        # Should find something reasonably close
        assert best_fitness > -0.3  # Within 0.3 of target
