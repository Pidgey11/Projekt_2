import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def podstawowe_sprawdzenia():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print(df_train.columns)  # wyswietlenie wszystkich kategorii
    print("Ilosc pustych miejsc", df_train.isnull().sum().sum())  # sprawdzenie ilości pustych miejsc


def wykresy():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_train = df_train.drop("id", axis=1)  # usuniecie id z wykresow
    df_train['Driving_License'] = df_train['Driving_License'].astype('object')
    df_train['Previously_Insured'] = df_train['Previously_Insured'].astype('object')
    df_train['Response'] = df_train['Response'].astype('object')
    df_num = df_train.select_dtypes(exclude='object')
    df_cat = df_train.select_dtypes(include='object')
    for i in df_num.columns:
        sns.distplot(df_num[i])
        plt.show()


if __name__ == '__main__':
    podstawowe_sprawdzenia()
    wykresy()
    ####

