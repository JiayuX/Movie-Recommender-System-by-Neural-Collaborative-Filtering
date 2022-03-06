import math
import numpy as np
import pandas as pd
import torch


def evaluate_hr(top_indices, true_item_idx):

	if true_item_idx in top_indices:
		return 1.
	else:
		return 0.


def evaluate_ndcg(top_indices, true_item_idx, gpu):

	if true_item_idx in top_indices:
		if gpu:
			return 1. / np.log2(top_indices.cpu().numpy().tolist().index(true_item_idx) + 2.)
		else:
			return 1. / np.log2(top_indices.numpy().tolist().index(true_item_idx) + 2.)
	else:
		return 0.
