import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

def main():
    label_column = 'quality'
    raw_data = pd.read_csv(r'./data/red_wine.csv')
    
    train_data, test_data = raw_data.iloc[:1450], raw_data.iloc[1450:]
    train_points, train_labels = train_data.drop([label_column], axis=1), train_data[[label_column]].copy()
    test_points, test_labels = test_data.drop([label_column], axis=1), test_data[[label_column]].copy()

    model = linear_model.LinearRegression()
    model.fit(train_points.values, train_labels.values)

    train_labels['prediction'] = model.predict(train_points)
    train_labels['prediction'] = train_labels['prediction'].apply(round)
    train_labels['correct'] = np.where(train_labels[label_column] == train_labels['prediction'], 1, 0)
    
    accuracy = train_labels['correct'].sum() / len(train_labels)
    corr = train_labels.drop(['correct'], axis=1).corr()
    print(f'Training Accuracy: {accuracy}')
    print(f'Training Correlation: {corr}')

    test_labels['prediction'] = model.predict(test_points)
    test_labels['prediction'] = test_labels['prediction'].apply(round)
    test_labels['correct'] = np.where(test_labels[label_column] == test_labels['prediction'], 1, 0)
    
    accuracy = test_labels['correct'].sum() / len(test_labels)
    corr = test_labels.drop(['correct'], axis=1).corr()
    print(f'Testing Accuracy: {accuracy}')
    print(f'Testing Correlation: {corr}')

    # train_labels['prediction'] = train_points.apply(lambda p: model.predict(p.values.reshape((1, ))))

if __name__ == "__main__":
    main()