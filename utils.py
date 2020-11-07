import numpy as np


def smooth_curve(x: np.ndarray, window: int = 30, sigma: float = 5) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    kernel = np.exp(-((np.arange(window) - window // 2) ** 2) / (2 * sigma ** 2))
    return np.convolve(x, kernel, "same") / np.convolve(np.ones_like(x), kernel, "same")


def max_diff(x: List) -> float:
    return max(x) - min(x)
