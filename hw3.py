import gzip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.neural_network import MLPClassifier

##READ DATA##

train_x, train_y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte')

test_x, test_y = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte', 
        labels_path='t10k-labels-idx1-ubyte')

##READ DATA##

##SCALE##

train_x[train_x > 0] = 1
train_x[train_x > 0] = 1

##SCALE##


##TESTS##

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
#print(test_y)

img_train_x = np.reshape(train_x,(60000,28,28))
img_test_x = np.reshape(test_x,(10000,28,28))
#plt.imshow(img_train_x[12],cmap=plt.cm.binary)
#plt.imshow(img_x[0])
#plt.show()

##TESTS##

##CLASSIFY##
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)

accuracies_relu_100_50 = []
accuracies_relu_300_100 = []

accuracies_relu_100_50_loss = []
accuracies_relu_300_100_loss = []



accuracies_tanh_100_50 = []
accuracies_tanh_300_100 = []

accuracies_tanh_100_50_loss = []
accuracies_tanh_300_100_loss = []

for i in range(1, 11):
    clf = MLPClassifier(hidden_layer_sizes=(100,50),solver='sgd',max_iter=i*10)
    #print(clf)
    clf.fit(train_x, train_y)
    print("Max iter ", i*10, "done!")
    print("Validation Score: ", clf.loss_)
    print("Prediction Score: ", clf.score(test_x, test_y))
    accuracies_relu_100_50.append(clf.score(test_x, test_y))
    accuracies_relu_100_50_loss.append(clf.loss_)

for i in range(1, 11):
    clf = MLPClassifier(hidden_layer_sizes=(300,100),solver='sgd',max_iter=i*10)
    #print(clf)
    clf.fit(train_x, train_y)
    print("Max iter ", i*10, "done!")
    print("Validation Score: ", clf.loss_)
    print("Prediction Score: ", clf.score(test_x, test_y))
    accuracies_relu_300_100.append(clf.score(test_x, test_y))
    accuracies_relu_300_100_loss.append(clf.loss_)
    

for i in range(1, 11):
    clf = MLPClassifier(hidden_layer_sizes=(100,50),solver='sgd',max_iter=i*10,activation='tanh')
    #print(clf)
    clf.fit(train_x, train_y)
    print("Max iter ", i*10, "done!")
    print("Validation Score: ", clf.loss_)
    print("Prediction Score: ", clf.score(test_x, test_y))
    accuracies_tanh_100_50.append(clf.score(test_x, test_y))
    accuracies_tanh_100_50_loss.append(clf.loss_)

for i in range(1, 11):
    clf = MLPClassifier(hidden_layer_sizes=(300,100),solver='sgd',max_iter=i*10,activation='tanh')
    #print(clf)
    clf.fit(train_x, train_y)
    print("Max iter ", i*10, "done!")
    print("Validation Score: ", clf.loss_)
    print("Prediction Score: ", clf.score(test_x, test_y))
    accuracies_tanh_300_100.append(clf.score(test_x, test_y))
    accuracies_tanh_300_100_loss.append(clf.loss_)

##CLASSIFY##

##FILE##
out0 = open("accuracies_100_50_relu.txt", "w")
for line in accuracies_100_50_relu:
    out0.write(line)
    out0.write("\n")
outF.close()

out1 = open("loss_100_50_relu_loss.txt", "w")
for line in accuracies_100_50_relu_loss:
    out1.write(line)
    out1.write("\n")
outF.close()

out0 = open("accuracies_300_100_relu.txt", "w")
for line in accuracies_300_100_relu:
    out0.write(line)
    out0.write("\n")
outF.close()

out0 = open("accuracies_300_100_relu_loss.txt", "w")
for line in accuracies_300_100_relu_loss:
    out0.write(line)
    out0.write("\n")
outF.close()

out0 = open("accuracies_100_50_tanh.txt", "w")
for line in accuracies_100_50_tanh:
    out0.write(line)
    out0.write("\n")
outF.close()

out0 = open("accuracies_100_50_tanh_loss.txt", "w")
for line in accuracies_100_50_tanh_loss:
    out0.write(line)
    out0.write("\n")
outF.close()

out0 = open("accuracies_300_100_tanh.txt", "w")
for line in accuracies_300_100_tanh:
    out0.write(line)
    out0.write("\n")
outF.close()

out0 = open("accuracies_300_100_tanh_loss.txt", "w")
for line in accuracies_300_100_tanh_loss:
    out0.write(line)
    out0.write("\n")
outF.close()

##FILE##

##DRAW##
x_axis = []

for i in range(1, 11):
    x_axis.append(i)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("# of Epochs")

plt.plot(x_axis, accuracies_relu_100_50, label='Relu, 100,50 Neurons')
plt.plot(x_axis, accuracies_relu_300_100, label='Relu, 300,100 Neurons' )
plt.plot(x_axis, accuracies_tanh_100_50, label='Tanh, 100,50 Neurons')
plt.plot(x_axis, accuracies_tanh_300_100, label='Tanh, 300,100 Neurons')
plt.legend('Relu, 100,50 Neurons', 'Relu 300,100 Neurons', 'Tanh, 100,50 Neurons', 'Tanh, 300,100 Neurons', loc='best')
plt.show()
plt.savefig("first.png")
##DRAW##
