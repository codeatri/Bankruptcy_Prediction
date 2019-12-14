import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff


def main():
    dataset = arff.loadarff(open('3year.arff', 'r'))
    df = pd.DataFrame(dataset[0])
    x = df.drop(["class"], axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)

    train_df["class"] = y_train
    test_df["class"] = y_test

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")


if __name__ == '__main__':
    main()