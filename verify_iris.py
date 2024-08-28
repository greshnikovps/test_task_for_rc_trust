import torch
import onnx
from onnx2torch import convert
from z3 import *

# Load the ONNX model
onnx_model = onnx.load("iris_model.onnx")  # Load the ONNX model from file
model = convert(onnx_model)  # Convert the ONNX model to a PyTorch model


def get_weights_and_biases(model):
    """
    Extracts the weights and biases from the converted PyTorch model.

    Arguments:
    model -- The converted PyTorch model.

    Returns:
    A tuple containing the weights and biases for each layer of the model.
    """
    # Access the layers and extract weights and biases
    weights_1 = model._modules['fc1/Gemm'].weight.detach().numpy()
    bias_1 = model._modules['fc1/Gemm'].bias.detach().numpy()
    weights_2 = model._modules['fc2/Gemm'].weight.detach().numpy()
    bias_2 = model._modules['fc2/Gemm'].bias.detach().numpy()
    weights_3 = model._modules['fc3/Gemm'].weight.detach().numpy()
    bias_3 = model._modules['fc3/Gemm'].bias.detach().numpy()

    return (weights_1, bias_1), (weights_2, bias_2), (weights_3, bias_3)


def ReLU(x):
    """
    Implements the ReLU activation function for Z3.

    Arguments:
    x -- Input value or list of values.

    Returns:
    The ReLU-transformed value(s).
    """
    if isinstance(x, list):
        return [If(element > 0, element, 0) for element in x]
    else:
        return If(x > 0, x, 0)


def Linear(x, weights, bias):
    """
    Implements the linear layer computation (Wx + b) for Z3.

    Arguments:
    x -- Input list of Z3 Real variables.
    weights -- Weights matrix for the layer.
    bias -- Bias vector for the layer.

    Returns:
    The output list after applying the linear transformation.
    """
    output = [sum(x[i] * weights[j][i] for i in range(len(x))) + bias[j] for j in range(len(weights))]
    return output


def encode_network_in_z3(solver, model, input_var):
    """
    Encodes the neural network layers in Z3 using the provided weights, biases, and input variables.

    Arguments:
    solver -- Z3 solver instance.
    model -- The converted PyTorch model.
    input_var -- List of Z3 Real variables representing the input.

    Returns:
    The final output of the network encoded in Z3.
    """
    # Get weights and biases
    (weights_1, bias_1), (weights_2, bias_2), (weights_3, bias_3) = get_weights_and_biases(model)

    # Encode each layer using Z3 operations
    x = ReLU(Linear(input_var, weights_1, bias_1))
    x = ReLU(Linear(x, weights_2, bias_2))
    x = Linear(x, weights_3, bias_3)
    return x


def add_precondition(solver, data_point, epsilon):
    """
    Adds the precondition to the solver ensuring that the input variables are within an epsilon neighborhood of the original data point.

    Arguments:
    solver -- Z3 solver instance.
    data_point -- The original data point.
    epsilon -- The maximum allowed perturbation (epsilon).
    """
    for i in range(len(data_point)):
        solver.add(Real(f'x{i}') >= data_point[i] - epsilon)
        solver.add(Real(f'x{i}') <= data_point[i] + epsilon)


def add_postcondition1(solver, encoded_output, original_class):
    """
    Adds the postcondition to the solver ensuring that the predicted class does not change.

    Arguments:
    solver -- Z3 solver instance.
    encoded_output -- The output of the network encoded in Z3.
    original_class -- The original predicted class.
    """
    for i in range(len(encoded_output)):
        if i != original_class:
            solver.add(encoded_output[original_class] > encoded_output[i])


def add_postcondition(solver, encoded_output, original_class):
    """
    Adds the negation of the postcondition to the solver ensuring that the predicted class does not change.

    Arguments:
    solver -- Z3 solver instance.
    encoded_output -- The output of the network encoded in Z3.
    original_class -- The original predicted class.

    Returns:
    None
    """
    conditions = []
    for i in range(len(encoded_output)):
        if i != original_class:
            conditions.append(encoded_output[original_class] <= encoded_output[i])
    solver.add(Or(conditions))


def verify_robustness(model, data_point, epsilon):
    """
    Verifies the robustness of the model for a given data point and epsilon.

    Arguments:
    model -- The converted PyTorch model.
    data_point -- The original data point.
    epsilon -- The maximum allowed perturbation (epsilon).

    Returns:
    "Counterexample" and the counterexample if a counterexample is found (i.e., the network is not robust),
    "Verified" if the network is robust for the given epsilon.
    """
    solver = Solver()

    # Convert data point to Z3 Real variables
    input_vars = [Real(f'x{i}') for i in range(len(data_point))]

    # Add the precondition (input bounds)
    add_precondition(solver, data_point, epsilon)

    # Encode the network in Z3
    encoded_output = encode_network_in_z3(solver, model, input_vars)

    # Determine the original class
    original_class = torch.argmax(model(torch.tensor(data_point, dtype=torch.float32))).item()

    # Add the postcondition (output class should not change)
    add_postcondition(solver, encoded_output, original_class)

    # Check the constraints
    if solver.check() == sat:
        return "Counterexample", solver.model()
    else:
        return "Verified"

# running verification for Epsilons values 0.1 and 2
data_point = [-0.90068117,  1.01900435, -1.34022653, -1.3154443 ]
epsilon_values = [0.1, 2]
for epsilon in epsilon_values:
    result = verify_robustness(model, data_point, epsilon)
    print(f'Epsilon = {epsilon}: {result}')

def print_max_robust_eps(model, data_point, start=0.1, end=2.0, tolerance=0.0001):
    """
    Finds and prints the maximum epsilon value for which the model is still robust using binary search.

    Arguments:
    model -- The converted PyTorch model.
    data_point -- The original data point.
    start -- The starting epsilon value for the search (default is 0.1).
    end -- The ending epsilon value for the search (default is 2.0).
    tolerance -- The stopping criterion for the binary search (default is 0.0001).

    Returns:
    The maximum epsilon value for which the model is still robust.
    """
    max_verified_epsilon = 0  # Initialize the maximum verified epsilon

    while abs(end - start) >= tolerance:
        mid = (start + end) / 2  # Calculate the midpoint
        result = verify_robustness(model, data_point, mid)

        if result == "Verified":
            max_verified_epsilon = mid
            start = mid + tolerance  # Move the start to mid + tolerance to search the higher range
        else:
            end = mid - tolerance  # Move the end to mid - tolerance to search the lower range

    if max_verified_epsilon == 0:
        print("The model is not verified even for the minimal epsilon.")
    else:
        print(
            f"Epsilon = {max_verified_epsilon:.4f} is the maximal (with tolerance={tolerance}) epsilon value for which the model is still robust.")

# running verification for finding maximal Epsilon value (with a certain error associated with discretization)
print_max_robust_eps(model, data_point, start=0.1, end=2.0)