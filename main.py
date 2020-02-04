import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
                im = Image.open('digits/' + str(i) + '/' + str(j) + '.png').convert('L')
                im_resized = im.resize(self.resolution, Image.ANTIALIAS)
                digit_list.append(im_resized)
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

        e = self.energy(prediction[0])

        for i in range(self.num_iter):
            prediction = np.sign(self.__weights @ prediction.T - self.threshold)
            e_new = self.energy(prediction.T[0])

            if e == e_new:
                print(e)
                return prediction
            prediction = prediction.T
            e = e_new
        print(e)
        return prediction

    def async_predict(self, path, random=False):
        img = Image.open(path).convert('L')
        im_resized = img.resize(self.resolution, Image.ANTIALIAS)

        np_img = self.__convert(im_resized)
        prediction = np_img
        e = self.energy(prediction[0])

        for i in range(self.num_iter):
            if random:
                for j in range(100):
                    idx = np.random.randint(0, self.resolution[0] * self.resolution[1])
                    prediction[0][idx] = np.sign(self.__weights[idx].T @ prediction[0] - self.threshold)
            else:
                for idx in range(0, self.resolution[0] * self.resolution[1]):
                    prediction[0][idx] = np.sign(self.__weights[idx].T @ prediction[0] - self.threshold)
            # Compute new state energy
            e_new = self.energy(prediction[0])

            # s is converged
            if e == e_new:
                print(e)
                return prediction
            # Update energy
            e = e_new
        print(e)
        return prediction

    def __convert(self, img):
        np_img = np.array(img, dtype='int64')
        np_img = ~np_img  # invert B&W
        np_img[np_img != -1] = 1
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


network = NeuralNetwork(1, (56, 56), 100)
network.train()
network.show_weights()
c = network.async_predict('digits/3/0.png', random=False).reshape((56, 56))
show_smth(c)
