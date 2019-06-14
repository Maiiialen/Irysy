# wczytywanie bibliotek
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np

# wczytanie danych o irysach
iris = datasets.load_iris()

# wczytanie wielkości (x) i nazwy kwiatu, który odpowiada wielkościom (y)
X = iris['data']
y = iris['target']

# ustawienie, że 30% danych będzie danymi testowymi, a 70% danymi uczącyni
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# trenowanie Scaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# utworzenie tablic z danymi do zmiany ustawień działania klasyfikacji
activ = ['relu', 'tanh']    # funkcje aktywacji
solv = ['adam', 'sgd']      # algorytmy uczenia
size = [5, 10, 50, 100]     # liczba warstw


# otworzenie pliku
plik = open('multi.txt', 'w')
try:
    for i in range(0, 2):               # zmiana funkcji aktywacji
        plik.write(activ[i])            # zapisanie tego do pliku
        plik.write("\n")
        for j in range(0, 2):           # zmiana algorytmu uczenia
            plik.write(solv[j])         # zapisanie tego do pliku
            for k in range(0, 4):           # zmiana ilości warstw
                plik.write("\n")    
                plik.write(str(size[k]))    # zapisanie tego do pliku
                plik.write("\n")
                for l in range(0, 100):     # powtórzenie prób sto razy
                    mlp = MLPClassifier(hidden_layer_sizes=(size[k],), max_iter=1000, solver=solv[j], activation=activ[i])  # stworzenie wielowarstwowego perceptrona
                    mlp.fit(X_train_std, y_train)       # uczenie
                    y_pred = mlp.predict(X_test_std)    # klasyfikacja
                    plik.write(str(accuracy_score(y_test, y_pred)))     # zapisa danych do pliku
                    plik.write(" ")
                plik.write("\n")
            plik.write("\n")
finally:
	plik.close()    #zamknięcie pliku
