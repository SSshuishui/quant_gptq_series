from re import L
import numpy as np
from pyparsing import line
import torch
from .binary import high_order_residual


def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse


def calculate_percentage_and_variance_original(weights, abs_weights, bin_edges):
    percentages = []
    variances = []
    accum_percentages = [0]
    total_elements = abs_weights.numel()
    for i in range(len(bin_edges) - 1):
        bin_mask = (abs_weights >= bin_edges[i]) & (abs_weights < bin_edges[i + 1])
        bin_weights = weights[bin_mask]
        percentages.append(bin_weights.numel() / total_elements * 100)
        accum_percentages.append(accum_percentages[-1] + percentages[-1])
        variances.append(torch.var(bin_weights))
    return percentages, variances, accum_percentages

'''
Include main method to search the rate for 2-bit salient data columns and the optimal split for 1-bit data
'''
def structural_searching(origin_matrix, up_lim=30):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)

        group4 = high_order_residual(origin_matrix, ~mask3, order=2)
        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)
        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)

    # search for the optimal split for the second group, high order=1,, non-structured search
    for split_value in percentile_values:
        mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
        group1 = high_order_residual(origin_matrix, mask1, order=1)
        group2 = high_order_residual(origin_matrix, mask2, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value
        tmp = torch.max(torch.abs(search_matrix)).item()
    
    return optimal_split, mask3


def find_optimal_split(group_max, origin_matrix, border):
    optimal_split = None
    minimal_value = float('inf')
    searching_steps = torch.arange(0.1,0.8,0.01)
    searching_steps = searching_steps * group_max

    group3 = high_order_residual(origin_matrix, torch.abs(origin_matrix) > border, order=2)
    for split_value in searching_steps:

        group1 = high_order_residual(origin_matrix, (torch.abs(origin_matrix) > split_value) & (torch.abs(origin_matrix) <= border), order=1)
        group2 = high_order_residual(origin_matrix, torch.abs(origin_matrix) <= split_value, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value

    return optimal_split, minimal_value


'''
Generate the structural mask on the basis of the split border
'''
def generate_structural_mask(origin_matrix, mask3, braq1_border):
    mask1_2 = ~mask3

    binary_group = torch.abs(origin_matrix*mask1_2)

    mask2 = binary_group >= braq1_border
    mask1 = binary_group < braq1_border

    mask1 = mask1 * mask1_2
    mask2 = mask2 * mask1_2

    return mask1, mask2


def generate_mask(origin_matrix, braq2_border, braq1_border):
    mask3 = torch.abs(origin_matrix) >= braq2_border
    mask1 = torch.abs(origin_matrix) <= braq1_border
    mask2 = (torch.abs(origin_matrix) > braq1_border) & (torch.abs(origin_matrix) < braq2_border)
    return mask1, mask2, mask3


'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=30):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3
