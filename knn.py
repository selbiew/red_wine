import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    target_column = 'quality'
    
    raw_data = pd.read_csv("./data/red_wine.csv")
    processed_data = raw_data.fillna(raw_data.mean())
    processed_data.loc[:, processed_data.columns != target_column] = processed_data.loc[:, processed_data.columns != target_column].apply(minmax_normalize, axis=0)

    train_data, validation_data = processed_data.sample(frac=0.8), processed_data.sample(frac=0.2) 
    
    k = optimum_k(train_data, validation_data, target_column, 1, 15)

    train_data, test_data = processed_data.sample(frac=0.9), processed_data.sample(frac=0.1)
    train_points, train_targets = train_data.drop([target_column], axis=1), train_data[[target_column]]
    test_points, test_targets = test_data.drop([target_column], axis=1), test_data[[target_column]]
    
    results = pd.DataFrame()
    results['prediction'] = test_points.apply(lambda p: knn(p.values, train_points.values, train_targets.values, k), axis=1)[0]
    results[target_column] = test_targets
    results['Correct'] = np.where(results[target_column] == results['prediction'], 1, 0)
    
    print(results.head())
    accuracy = results['Correct'].sum() / len(results)
    print(f'Accuracy: {accuracy}')
    plt.show()
    
def optimum_k(train_data, validation_data, target_column, start, stop, step=2, weight_function=lambda d, t: t):
    accuracies = []
    train_points, train_targets = train_data.drop([target_column], axis=1), train_data[[target_column]]
    validation_points, validation_targets = validation_data.drop([target_column], axis=1), validation_data[[target_column]]

    for k in range(start, stop + 1, step):
        results = pd.DataFrame()
        results['prediction'] = validation_points.apply(lambda p: knn(p.values, train_points.values, train_targets.values, k, weight_function), axis=1)[0]
        results[target_column] = validation_targets
        results['Correct'] = np.where(results[target_column] == results['prediction'], 1, 0)
        
        accuracies.append((k, results['Correct'].sum() / len(results)))
    
    accuracies.sort(key=lambda t: t[1], reverse=True)
    return accuracies[0][0]

def knn(point, points, targets, k=7, weight_function=lambda d, t: t):
    neighbours = [(i, np.linalg.norm(point - neighbour)) for i, neighbour in enumerate(points)]
    ksmallest_distance = quickselect([n[1] for n in neighbours], 0, len(neighbours) - 1, k)
    neighbour_predictions = [weight_function(distance, targets[index]) for index, distance in neighbours if distance <= ksmallest_distance]
    
    return round(pd.DataFrame(neighbour_predictions).mean())

def minmax_normalize(column):
    column_min, column_max = column.min(), column.max()
    return column.apply(lambda v: (v - column_min) / (column_max - column_min))

def quickselect(array, left_index, right_index, k):
    if left_index == right_index:
        return array[left_index]
    pivot_index = random.randint(left_index, right_index)
    pivot_index = partition(array, left_index, right_index, pivot_index)
    if k == pivot_index:
        return array[k]
    elif k < pivot_index:
        return quickselect(array, left_index, pivot_index - 1, k)
    return quickselect(array, pivot_index + 1, right_index, k)

def partition(array, left_index, right_index, pivot_index):
    pivot_value = array[pivot_index]
    array[pivot_index], array[right_index] = array[right_index], array[pivot_index]
    store_index = left_index
    for i in range(left_index, right_index):
        if array[i] < pivot_value:
            array[store_index], array[i] = array[i], array[store_index]
            store_index += 1
    array[right_index], array[store_index] = array[store_index], array[right_index]

    return store_index

if __name__ == "__main__":
    main()