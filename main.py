import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from tqdm import tqdm


def show_smth(s):
    plt.imshow(s)
    plt.colorbar()
    plt.show()


class NeuralNetwork:
    resolution = (0, 0)
    __weights = np.array([])
    __digits = {}
    number_of_digits = 1
    threshold = 0
    num_iter = 0

    def __init__(self, number_of_digits=1, resolution=(0, 0), iterations=0):
        self.num_iter = iterations
        self.resolution = resolution
        for i in range(10):
            digit_list = []
            for j in range(number_of_digits):
                digit_list.append(Image.open('digits/' + str(i) + '/' + str(j) + '.png').convert('L'))
            self.__digits[i] = digit_list

    def train(self):
        self.__weights = np.zeros((self.resolution[0] * self.resolution[1], self.resolution[0] * self.resolution[1]))
        for digit in self.__digits:
            for img in self.__digits[digit]:
                np_img = self.__convert(img)
                self.__weights += np_img.T @ np_img
        np.fill_diagonal(self.__weights, 0)

    def train_weights(self):
        train_data = []
        for digit in self.__digits:
            for img in self.__digits[digit]:
                np_img = self.__convert(img)
                train_data.append(np_img[0])

        print("Start to train weights...")

        num_data = len(train_data)

        # initialize weights
        W = np.zeros((self.resolution[0] * self.resolution[1], self.resolution[0] * self.resolution[1]))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.resolution[0] * self.resolution[1])

        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)

        # Make diagonal element of W into 0
        np.fill_diagonal(W, 0)
        W /= num_data

        self.__weights = W

    def show_weights(self):
        show_smth(self.__weights)

    def predict(self, path):
        img = Image.open(path).convert('L')
        np_img = self.__convert(img)
        prediction = np_img

        e = self.energy(np_img[0])

        for i in range(self.num_iter):
            prediction = np.sign(self.__weights @ prediction.T - self.threshold)
            e_new = self.energy(prediction.T[0])

            if e == e_new:
                return prediction
            prediction = prediction.T
            e = e_new
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

    def __convert_to_float(self, img):
        np_img = np.array(img, dtype='float')
        np_img = np_img.reshape((1, self.resolution[0] * self.resolution[1]))
        for i in range(len(np_img[0])):
            np_img[0][i] /= 256
        return np_img

    def sum(self):
        print(sum(sum(self.__weights)))

    def energy(self, s):
        return -0.5 * s @ self.__weights @ s + np.sum(s * self.threshold)


network = NeuralNetwork(1, (28, 28), 100)
network.train()
network.show_weights()
c = network.predict('digits/1/0.png').reshape((28, 28))
show_smth(c)
network.sum()
