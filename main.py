import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def podstawowe_sprawdzenia():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print(df_train.columns)  # wyswietlenie wszystkich kategorii
    print("Ilosc pustych miejsc", df_train.isnull().sum().sum())  # sprawdzenie ilości pustych miejsc


def podstawowe_statystyki():
    df_train = pd.read_csv('train.csv')
    age_min = df_train.Age.min()
    age_max = df_train.Age.max()
    age_sr = df_train.Age.sum() / df_train.Age.count()
    print("Sredni wiek =", age_sr)
    print("Max wiek = ", age_max)


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
def drzewo_decyzyjne():
    df_train = pd.read_csv('train.csv')
    df_train = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Damage'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Age'], drop_first=True)
    scaler = StandardScaler()
    inputs = scaler.fit_transform(df_train.drop('Response', axis=1))
    x = inputs
    y = df_train['Response']
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.3, random_state=101)
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    print("Wynik dla drzewa decyzyjnego:",dtc.score(x_test,y_test))


def naive_bayes():
    df_train = pd.read_csv('train.csv')
    NB = GaussianNB()
    df_train = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Damage'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Age'], drop_first=True)
    scaler = StandardScaler()
    inputs = scaler.fit_transform(df_train.drop('Response', axis=1))
    x = inputs
    y = df_train['Response']
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.3, random_state=101)
    NB.fit(x_train,y_train)
    print(NB.score(x_test,y_test))

def sasiedzi():
    df_train = pd.read_csv('train.csv')
    df_train = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Damage'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Age'], drop_first=True)
    scaler = StandardScaler()
    inputs = scaler.fit_transform(df_train.drop('Response', axis=1))
    x = inputs
    y = df_train['Response']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
    KNN = KNeighborsClassifier()
    KNN.fit(x_train, y_train)
    print(KNN.score(x_test,y_test))

def neural():
    df_train = pd.read_csv('train.csv')
    df_train = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Damage'], drop_first=True)
    df_train = pd.get_dummies(df_train, columns=['Vehicle_Age'], drop_first=True)
    scaler = StandardScaler()
    inputs = scaler.fit_transform(df_train.drop('Response', axis=1))
    x = inputs
    y = df_train['Response']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
    clf = MLPClassifier()
    clf.fit(x_train,y_train)
    print(clf.score(x_test,y_test))

if __name__ == '__main__':
    podstawowe_sprawdzenia()
    drzewo_decyzyjne()
    podstawowe_statystyki()
    wykresy()

