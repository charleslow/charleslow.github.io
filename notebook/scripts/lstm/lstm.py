import time

import torch
import torch.nn as nn

# Parameters
input_size = 256
hidden_size = 256
num_layers = 2
batch_size = 500
seq_len = 10
num_tests = 10


def check_multi_pass_equals_single_pass(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
):
    model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    input_sequence = torch.randn(batch_size, seq_len, input_size)

    # Single pass
    output_all, hidden_all = model(input_sequence)

    # 2 passes
    midpoint = seq_len // 2
    first_half = input_sequence[:, :midpoint, :]
    output_first, hidden_first = model(first_half)
    second_half = input_sequence[:, midpoint:, :]
    output_second, hidden_second = model(second_half, hidden_first)

    print(torch.allclose(output_all[:, -1, :], output_second[:, -1, :]))


def check_prediction_speed(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_tests: int,
):
    model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    times = []
    for _ in range(num_tests):
        input_sequence = torch.randn(batch_size, seq_len, input_size)
        with torch.no_grad():
            start_time = time.time()
            output, hidden = model(input_sequence)
            end_time = time.time()
        times.append(end_time - start_time)
    avg_time_per_pass = sum(times) / len(times)
    print(f"Average time per forward pass: {avg_time_per_pass * 1000:.0f} milliseconds")


if __name__ == "__main__":
    # We can pre-compute hidden state over each document, then at inference time
    # run the second pass over the query
    check_multi_pass_equals_single_pass(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    # We can rerank 500 examples in this way in around 100ms on CPU
    check_prediction_speed(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        num_tests=num_tests,
    )
