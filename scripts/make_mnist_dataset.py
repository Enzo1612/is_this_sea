from ..src.dataloader import MNIST_DataLoader

dataLoader = MNIST_DataLoader()
X, y = dataLoader.load_mnist()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = dataLoader.get_binary_target(y_train, '5')




