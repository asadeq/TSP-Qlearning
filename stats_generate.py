import numpy as np
from scipy import stats

def analyze_times(times, confidence=0.95):
    """
    Compute the mean and confidence interval for a set of measurements.

    Parameters:
        times (list or array): Running time measurements.
        confidence (float): Confidence level (default 0.95).

    Returns:
        mean (float), lower_bound (float), upper_bound (float)
    """
    n = len(times)
    mean = np.mean(times)
    sem = stats.sem(times)  # Standard error of the mean
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * sem
    return mean, mean - margin, mean + margin

if __name__ == "__main__":
    # Example: user inputs measurements separated by spaces
    data = input("Enter running times separated by spaces: ")
    times = [float(x) for x in data.split()]
    
    mean, lower, upper = analyze_times(times)
    
    print(f"\nMean running time: {mean:.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")
