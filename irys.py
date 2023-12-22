from sklearn import datasets

# Załaduj zbiór danych iris
iris = datasets.load_iris()

# Cechy (wymiary) kwiatów
features = iris.data
# features = [[1,2,3,9],[2,4,5,0]]
# Etykiety klas (gatunki irysów)
labels = iris.target
from sklearn.model_selection import train_test_split

# Podziel zbiór na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)
from sklearn.neighbors import KNeighborsClassifier

# Utwórz model klasyfikatora k-najbliższych sąsiadów
knn = KNeighborsClassifier(n_neighbors=3)

# Naucz model na danych treningowych
knn.fit(X_train, y_train)
# Przewiduj klasy na danych testowych
predictions = knn.predict(X_test)
from sklearn import metrics

# Porównaj przewidywane klasy z rzeczywistymi klasami
accuracy = metrics.accuracy_score(y_test, predictions)
print("Dokładność:", accuracy)
# print(iris.data)
