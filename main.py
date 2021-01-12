import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print(df_train.isnull().sum()) # sprawdzenie ilości pustych miejsc
    Gender = {'Male': 1, 'Female': 0}
    df_train.Gender = [Gender[item] for item in df_train.Gender]
    plt.hist(df_train.Gender,  edgecolor='black')
    plt.show()
