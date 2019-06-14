# wczytywanie bibliotek
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import numpy as np

# wczytanie danych o irysach
iris = datasets.load_iris()

# wczytanie wielkości (x) i nazwy kwiatu, który odpowiada wielkościom (y)
X = iris['data']
y = iris['target']

# trenowanie Scaler
sc = StandardScaler()

# utworzenie tablicy określającej jaka część ma zostać przeznaczona do testu
size = [0.3, 0.5, 0.7]

# otworzenie pliku
plik = open('perceptron.txt', 'w')
try:
    plik.write('perceptron ')    # podpisanie danych w pliku
    plik.write("\n")
    for j in range(0, 3):   # zmiana wielkości określająca jaka część danych ma zostać podana do uczenia, a jaka do testu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size[j])    
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        plik.write('perceptron' + str(size[j]) + ' ')    # podpisanie danych w pliku
        for i in range(0, 100):     # test powtórzony 100 razy
            ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)     # tworzenie perceptronu
            ppn.fit(X_train_std, y_train)       # uczenie
            y_pred = ppn.predict(X_test_std)    # klasyfikowanie
            plik.write(str(accuracy_score(y_test, y_pred)))     # zapis do pliku
            plik.write(" ")
        plik.write("\n")
finally:
	plik.close()    # zamknięcie pliku
