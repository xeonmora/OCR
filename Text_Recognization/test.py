import os

import cv2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, \
    LSTM
from tensorflow.keras.models import Model

char_list = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6',
             '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '`', 'a',
             'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
             'w', 'x', 'y', 'z', '{', '|', '}', '~']

# input with shape of height=32 and width=128
inputs = Input(shape=(32, 128, 1))

conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(64, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

act_model = Model(inputs, outputs)


def pre_process_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    if h < 32:
        add_zeros = np.ones((32 - h, w)) * 255
        img = np.concatenate((img, add_zeros))
        h = 32

    if w < 128:
        add_zeros = np.ones((h, 128 - w)) * 255
        img = np.concatenate((img, add_zeros), axis=1)
        w = 128

    if w > 128 or h > 32:
        img = cv2.resize(img, (128, 32))

    img = np.expand_dims(img, axis=2)

    # Normalize each image
    img = img / 255.

    return img


act_model.load_weights('./Text_Recognization/C_LSTM_best.hdf5')


def predict_output(img):
    # predict outputs on validation images
    prediction = act_model.predict(np.array([img]))
    char = []
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction,
                                   input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])
    for x in out:
        for p in x:

            if int(p) != -1:
                char.append(char_list[int(p)])

    return "".join(char)


def text_recognize():
    final_text = ""

    b = [float(sent.split(".jpg")[0]) for sent in os.listdir('./Text_Recognization/demo_images')]
    b.sort()

    for i in range(0, len(b)):
        img_name = str(b[i]) + ".jpg"
        test_img = pre_process_image('./Text_Recognization/demo_images/' + img_name)

        if i != len(b) - 1 and int(int(b[i]) - int(b[i + 1])) == -1:
            final_text = final_text + " " + predict_output(test_img) + "\n"
        else:
            final_text = final_text + " " + predict_output(test_img)

    return final_text


if __name__ == '__main__':
    text_recognize()
