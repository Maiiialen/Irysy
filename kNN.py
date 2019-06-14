# wczytywanie bibliotek
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# wczytanie danych o irysach
iris = datasets.load_iris()

# wczytanie wielkości (x) i nazwy kwiatu, który odpowiada wielkościom (y)
X = iris['data']
y = iris['target']

# trenowanie Scaler
sc = StandardScaler()

# utworzenie tablicy określającej jaka część ma zostać przeznaczona do testu
size = [0.3, 0.5, 0.7]


# wykonanie dla NN
# otworzenie pliku
plik = open('NN.txt', 'w')
try:
    plik.write('NN')    # podpisanie danych w pliku
    for j in range(0, 3):   # zmiana wielkości określająca jaka część danych ma zostać podana do uczenia, a jaka do testu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size[j])
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        plik.write('NN' + str(size[j]) + ' ')    # podpisanie danych w pliku
        for i in range(0, 100):     # test powtórzony 100 razy
            knn = KNeighborsClassifier(n_neighbors=1)   # tworzenie rozpoznawania jednego najbliższego sąsiada
            knn.fit(X_train_std, y_train)       # uczenie
            y_pred = knn.predict(X_test_std)    # klasyfikowanie
            plik.write(str(accuracy_score(y_test, y_pred)))    # wpisanie danych w pliku
            plik.write(" ")
        plik.write("\n")
finally:
	plik.close()


# wykonanie dla kNN z k=3
# otworzenie pliku
plik = open('3NN.txt', 'w')
try:
    plik.write('3NN')    # podpisanie danych w pliku
    for j in range(0, 3):   # zmiana wielkości określająca jaka część danych ma zostać podana do uczenia, a jaka do testu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size[j])
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        plik.write('3NN' + str(size[j]) + ' ')    # podpisanie danych w pliku
        for i in range(0, 100):     # test powtórzony 100 razy
            knn = KNeighborsClassifier(n_neighbors=3)   # tworzenie rozpoznawania dla 3 najbliższych sąsiadów
            knn.fit(X_train_std, y_train)       # uczenie
            y_pred = knn.predict(X_test_std)    # klasyfikowanie
            plik.write(str(accuracy_score(y_test, y_pred)))    # wpisanie danych w pliku
            plik.write(" ")
        plik.write("\n")
finally:
	plik.close()


# wykonanie dla kNN z k=5
# otworzenie pliku
plik = open('5NN.txt', 'w')
try:
    plik.write('5NN')    # podpisanie danych w pliku
    for j in range(0, 3):   # zmiana wielkości określająca jaka część danych ma zostać podana do uczenia, a jaka do testu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size[j])
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        plik.write('5NN' + str(size[j]) + ' ')    # podpisanie danych w pliku
        for i in range(0, 100):     # test powtórzony 100 razy
            knn = KNeighborsClassifier(n_neighbors=5)   # tworzenie rozpoznawania dla 5 najbliższych sąsiadów
            knn.fit(X_train_std, y_train)       # uczenie
            y_pred = knn.predict(X_test_std)    # klasyfikowanie
            plik.write(str(accuracy_score(y_test, y_pred)))    # wpisanie danych w pliku
            plik.write(" ")
        plik.write("\n")
finally:
	plik.close()