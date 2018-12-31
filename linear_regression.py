import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    target_column = 'quality'
    raw_data = pd.read_csv(r'./data/red_wine.csv')
    raw_data['bias'] = 1.0

    train_data = raw_data.iloc[:1400, :]
    test_data = raw_data.iloc[1400:, :]
    test_features = test_data.drop(target_column, 1)
    test_targets = test_data[[target_column]]

    weights = gradient_descent(train_data, target_column, batch_size=len(train_data), epochs=1000)
    hypotheses = np.rint(np.matmul(test_features, weights))
    sse = calculate_sse(hypotheses, test_targets)
    ase = sse / len(hypotheses)
    print(f'SSE: {sse}\nASE: {ase}')

    test_targets.hist()
    pd.DataFrame(hypotheses).hist()

    plt.show()

def gradient_descent(data, target_column, batch_size=1, learning_rate=0.0000001, epochs=10000):
    weights = np.zeros((data.shape[1] - 1, 1))

    for _ in range(epochs):
        sample = data.sample(n=batch_size)
        features = sample.drop(target_column, 1)
        targets = sample[[target_column]]
        hypotheses = np.matmul(features, weights)
        gradient = np.matmul(features.T, hypotheses - targets)
        weights -= learning_rate * gradient

    return weights

def calculate_sse(hypotheses, targets):
    return np.matmul(hypotheses.T - targets.T, hypotheses - targets)

if __name__ == '__main__':
    main()