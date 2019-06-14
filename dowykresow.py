# wczytywanie bibliotek
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# wczytanie danych o irysach
iris = datasets.load_iris()

# wczytanie wielkości (x) i nazwy kwiatu, który odpowiada wielkościom (y)
X = iris['data']
y = iris['target']

# trenowanie Scaler
sc = StandardScaler()


# wykresy
zmienna = plt.figure(1)
a = zmienna.add_subplot(231)
b = zmienna.add_subplot(232)
c = zmienna.add_subplot(233)
d = zmienna.add_subplot(234)
e = zmienna.add_subplot(235)
f = zmienna.add_subplot(236)

x1 = X[0:49,0]      # szerokośc liścia przy kwiacie
y1 = X[0:49,1]      # długość liścia przy kwiacie
x2 = X[50:99,0]
y2 = X[50:99,1]
x3 = X[100:149,0]
y3 = X[100:149,1]
a.plot(x1, y1, 'ro')    # wyswietlanie
a.plot(x2, y2, 'bs')
a.plot(x3, y3, 'g^')
a.grid(True)
a.set_xlabel("szerokośc liścia przy kwiacie [cm]")
a.set_ylabel("długość liścia przy kwiacie [cm]")
a.set_title("wykres zaleznosci wielkosci liscia przy kwiecie")

x1 = X[0:49,0]      # szerokośc liścia przy kwiacie
y1 = X[0:49,2]      # szerokosc platka kwiatu
x2 = X[50:99,0]
y2 = X[50:99,2]
x3 = X[100:149,0]
y3 = X[100:149,2]
b.plot(x1, y1, 'ro')
b.plot(x2, y2, 'bs')
b.plot(x3, y3, 'g^')
b.grid(True)
b.set_xlabel("szerokośc liścia przy kwiacie [cm]")
b.set_ylabel("szerokosc platka kwiatu [cm]")
b.set_title("wykres zaleznosci wielkosci liscia przy kwiecie")

x1 = X[0:49,0]      # szerokośc liścia przy kwiacie
y1 = X[0:49,3]      # długość platka kwiatu
x2 = X[50:99,0]
y2 = X[50:99,3]
x3 = X[100:149,0]
y3 = X[100:149,3]
c.plot(x1, y1, 'ro')
c.plot(x2, y2, 'bs')
c.plot(x3, y3, 'g^')
c.grid(True)
c.set_xlabel("szerokośc liścia przy kwiacie [cm]")
c.set_ylabel("długość platka kwiatu [cm]")
c.set_title("wykres zaleznosci wielkosci liscia przy kwiecie")

x1 = X[0:49,1]      # długość liścia przy kwiacie
y1 = X[0:49,2]      # szerokosc platka kwiatu
x2 = X[50:99,1]
y2 = X[50:99,2]
x3 = X[100:149,1]
y3 = X[100:149,2]
d.plot(x1, y1, 'ro')
d.plot(x2, y2, 'bs')
d.plot(x3, y3, 'g^')
d.grid(True)
d.set_xlabel("długość liścia przy kwiacie [cm]")
d.set_ylabel("szerokosc platka kwiatu [cm]")
d.set_title("wykres zaleznosci wielkosci platka kwiatu")

x1 = X[0:49,1]      # długość liścia przy kwiacie
y1 = X[0:49,3]      # długość platka kwiatu
x2 = X[50:99,1]
y2 = X[50:99,3]
x3 = X[100:149,1]
y3 = X[100:149,3]
e.plot(x1, y1, 'ro')
e.plot(x2, y2, 'bs')
e.plot(x3, y3, 'g^')
e.grid(True)
e.set_xlabel("długość liścia przy kwiacie [cm]")
e.set_ylabel("długość platka kwiatu [cm]")
e.set_title("wykres zaleznosci wielkosci platka kwiatu")

x1 = X[0:49,2]      # szerokosc platka kwiatu
y1 = X[0:49,3]      # długość platka kwiatu
x2 = X[50:99,2]
y2 = X[50:99,3]
x3 = X[100:149,2]
y3 = X[100:149,3]
f.plot(x1, y1, 'ro')
f.plot(x2, y2, 'bs')
f.plot(x3, y3, 'g^')
f.grid(True)
f.set_xlabel("szerokosc platka kwiatu [cm]")
f.set_ylabel("długość platka kwiatu [cm]")
f.set_title("wykres zaleznosci wielkosci platka kwiatu")

plt.show()
