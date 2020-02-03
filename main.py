import numpy as np
from PIL import Image
import mnist
import matplotlib.pyplot as plt
from random import randint


class NeuralNetwork:
    resolution = (0, 0)
    __weights = np.array([])
    __digits = {}
    number_of_digits = 1

    def __init__(self, number_of_digits=1, resolution=(0, 0)):
        self.resolution = resolution
        for i in range(0, 4):
            digit_list = []
            for j in range(number_of_digits):
                digit_list.append(Image.open('digits/' + str(i) + '/' + str(j) + '.png').convert('L'))
            self.__digits[i] = digit_list

    def train(self):
        self.__weights = np.zeros((self.resolution[0] * self.resolution[1], self.resolution[0] * self.resolution[1]))
        for digit in self.__digits:
            for img in self.__digits[digit]:
                np_img = self.__convert(img)
                self.__weights += np_img.T.dot(np_img)
        np.fill_diagonal(self.__weights, 0)

    def show_weights(self):
        plt.imshow(self.__weights)
        plt.colorbar()
        plt.show()

    def predict(self, path):
        img = Image.open(path).convert('L')
        np_img = self.__convert(img)
        prediction = self.__weights.dot(np_img.T)
        for i in range(len(prediction)):
            prediction[i] = sign(prediction[i])
        for j in range(100):
            prediction = self.__weights.dot(prediction)
            for i in range(len(prediction)):
                prediction[i] = sign(prediction[i])
        prediction = prediction.reshape(self.resolution)
        plt.imshow(prediction)
        plt.colorbar()
        plt.show()
        return prediction

    def async_predict(self, path):
        img = Image.open(path).convert('L')
        np_img = self.__convert(img)
        prediction = np_img.copy()
        for i in range(self.resolution[0] * self.resolution[0]):
            rand_column = randint(0, self.resolution[0] * self.resolution[0] - 1)
            prediction[0][i] = prediction.dot(self.__weights[i].T)
        # for i in range(len(prediction)):
        #     if prediction[0][i] > 0:
        #         prediction[0][i] = 1
        #     else:
        #         prediction[0][i] = -1
        prediction = prediction.reshape(self.resolution)
        plt.imshow(prediction)
        plt.colorbar()
        plt.show()
        return prediction

    def __convert(self, img):
        np_img = np.array(img, dtype='int64')
        np_img = ~np_img  # invert B&W
        np_img[np_img == 0] = -1
        np_img[np_img < -1] = 1
        np_img = np_img.reshape((1, self.resolution[0] * self.resolution[1]))
        return np_img

    def sum(self):
        print(sum(sum(self.__weights)))


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


network = NeuralNetwork(1, (28, 28))
network.train()
network.show_weights()
c = network.predict('mnist_png/testing/0/28.png')
network.sum()
