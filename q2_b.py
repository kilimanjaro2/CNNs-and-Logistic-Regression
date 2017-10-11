import cPickle as PIQL
import math
import os.path as op
import sys
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical as cg
from keras.callbacks import LearningRateScheduler

def unpickle(file):
    with open(file, 'rb') as lol:
        opened_file = PIQL.load(lol)
    return opened_file

def get_input_data():
    path = sys.argv[1]
    testpath = sys.argv[2]
    return path, testpath

def step_decay(epoch):
    epochs_drop = 10.0
    # lrate = (math.pow(0.8, math.floor((1+epoch)/epochs_drop)) * 0.001)
    # lrate = min(lrate,0.0001)
    #drop = 0.8
    #return lrate
    return min((math.pow(0.8, math.floor((1+epoch)/epochs_drop)) * 0.001),0.0001)

def pre_process(path, testpath):
    metadata = unpickle(op.join(path, "batches.meta"))
    label_names = metadata["label_names"]

    e = unpickle(op.join(path, "data_batch_5"))
    d = unpickle(op.join(path, "data_batch_4"))
    c = unpickle(op.join(path, "data_batch_3"))
    b = unpickle(op.join(path, "data_batch_2"))
    a = unpickle(op.join(path, "data_batch_1"))

    t = unpickle(testpath)

    a['data'] = np.reshape(a['data'], (10000, 32, 32, 3), order='F')
    b['data'] = np.reshape(b['data'], (10000, 32, 32, 3), order='F')
    c['data'] = np.reshape(c['data'], (10000, 32, 32, 3), order='F')
    d['data'] = np.reshape(d['data'], (10000, 32, 32, 3), order='F')
    e['data'] = np.reshape(e['data'], (10000, 32, 32, 3), order='F')
    t['data'] = np.reshape(t['data'], (10000, 32, 32, 3), order='F')

    x_train = np.concatenate((a['data'], b['data'], c['data'], d['data'], e['data']), axis=0)
    y_train = np.concatenate((cg(a['labels'], 10), cg(b['labels'], 10), cg(c['labels'], 10), cg(d['labels'], 10), cg(e['labels'], 10)), axis=0)

    x_test = t['data']
    y_test = cg(t['labels'], 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = (x_train - 80) / 255
    x_test = (x_test - 80) / 255

    return label_names, x_train, y_train, x_test, y_test

def bat_and_act(model):
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    return model

def build_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    pair = (3,3)
    model.add(Conv2D(32, pair, padding='same', input_shape=x_train.shape[1:]))
    model = bat_and_act(model)

    model.add(Conv2D(32, pair))
    model = bat_and_act(model)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, pair, padding='same'))
    bat_and_act(model)

    model.add(Conv2D(64, pair))
    model = bat_and_act(model)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(512))
    model = bat_and_act(model)
    model.add(Dropout(0.3))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.001))
    return model

def build_generator(x_train):
    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)

    datagen.fit(x_train)
    return datagen

def predict_output(model,x_test):
    return model.predict(x_test)

def print_op(label_names,op):
    f = open('q2_b_output.txt', 'w')
    for i in op:
        f.write(label_names[np.argmax(i)] + '\n')
    f.close()

if __name__ == '__main__':
    path , testpath = get_input_data()
    label_names, x_train, y_train, x_test, y_test = pre_process(path, testpath)
    model = build_model(x_train, y_train, x_test, y_test)
    x_train_size = x_train.shape[0]
    np.random.seed(10)
    batch_size = 128
    epochs = 50
    #number of classes = 10

    datagen = build_generator(x_train)
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    model.fit_generator(datagen.flow(x_train, y_train, shuffle = True,batch_size=128),
            steps_per_epoch=x_train_size // batch_size,
            verbose=1,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=(x_test, y_test),
            workers=4)

    op = predict_output(model,x_test)
    print_op(label_names,op)
