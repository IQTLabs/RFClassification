# scripts to measure the prediction latency for sklearn models
# Code from : https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html
import time
import gc
import numpy as np

def atomic_benchmark_estimator(estimator, X_test, output_type, verbose=False):
    """Measure runtime prediction of each instance."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    ys = np.empty(n_instances, dtype=output_type) # '<U3' for dronedetect data
    for i in range(n_instances):
        instance = X_test[[i], :]
        start = time.time()
        yi = estimator.predict(instance)
        
        runtimes[i] = time.time() - start
        ys[i] = yi[0]
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return ys, runtimes

def bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose):
    """Measure runtime prediction of the whole input."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_bulk_repeats, dtype=float)
    for i in range(n_bulk_repeats):
        start = time.time()
        estimator.predict(X_test)
        runtimes[i] = time.time() - start
    runtimes = np.array(list(map(lambda x: x / float(n_instances), runtimes)))
    if verbose:
        print(
            "bulk_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes