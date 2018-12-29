import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

def main():
    raw_data = pd.read_csv('./data/red_wine.csv')
    raw_data['bias'] = 1.0
    target_column = 'quality'

    train_data = raw_data.sample(frac=0.8, replace=True)
    test_data = raw_data.sample(frac=0.2, replace=True)
    test_features = test_data.drop(target_column, 1)
    test_targets = test_data[[target_column]]

    weights = gradient_descent(train_data, target_column, batch_size=len(train_data))
    hypotheses = np.rint(np.matmul(test_features, weights))
    sse = calculate_sse(hypotheses, test_targets)
    print(sse)

def gradient_descent(data, target_column, batch_size=1, learning_rate=0.0000001, epochs=10000):
    weights = np.random.rand(data.shape[1] - 1, 1)

    for _ in range(epochs):
        sample = data.sample(n=batch_size)
        features = sample.drop('quality', 1)
        targets = sample[['quality']]
        hypotheses = np.rint(np.matmul(features, weights))
        gradient = np.matmul(features.T, hypotheses - targets)
        weights -= learning_rate * gradient
    
    return weights

def calculate_sse(hypotheses, targets):
    return np.matmul(hypotheses.T - targets.T, hypotheses - targets)

if __name__ == "__main__":
    main()