import torch
import random
import numpy as np
import pandas as pd


def _safe_float(value):
    """
    Robustly converts an Excel cell value to np.float32.
    Returns None for invalid, unparseable, or empty/NaN values.
    """
    value = pd.to_numeric(value, errors='coerce')
    if pd.isna(value):
        return None
    return np.float32(value)


def replaceNone(lst) -> list:
    """
    Replaces 'None' values in a list using nearest neighbor imputation,
    prioritizing Last Observation Carried Forward (LOCF).
    """
    while None in lst:
        for idx_1 in range(len(lst)):
            if lst[idx_1] is None:
                # Find the nearest valid observation to the left (past data)
                left = next((lst[idx_2] for idx_2 in range(idx_1 - 1, -1, -1) if lst[idx_2] is not None), None)

                # Find the nearest valid observation to the right (future data)
                right = next((lst[idx_2] for idx_2 in range(idx_1 + 1, len(lst)) if lst[idx_2] is not None), None)

                # Prioritize past values (LOCF strategy aligns with chronological causality)
                if left is not None:
                    lst[idx_1] = left
                # If no past values exist, fallback to borrowing from the future
                elif right is not None:
                    lst[idx_1] = right

    return lst