import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

ALPHA_DICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L',
    20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}

class CharClassifier:
    def __init__(self, weights_path):
        self.reco_model = Sequential()
        self.reco_model.add(Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.reco_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.reco_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.reco_model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.reco_model.add(BatchNormalization())
        self.reco_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.reco_model.add(Flatten())
        self.reco_model.add(Dense(1024, activation='relu'))
        self.reco_model.add(Dropout(0.5))
        self.reco_model.add(Dense(31, activation='softmax'))

        self.reco_model.load_weights(weights_path)

    def predict(self, characters):
        if len(characters) == 0:
            raise Exception("characters is empty!")
        char_tensor = np.array(characters).reshape((len(characters), 28, 28, 1))
        char_tensor = char_tensor / 255.0

        result = self.reco_model.predict(char_tensor)
        result = np.argmax(result, axis=1)

        chars = [ALPHA_DICT[i] for i in result]

        return chars