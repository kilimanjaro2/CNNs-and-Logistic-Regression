import cPickle as PIQL
import math
import os.path as op
import sys
import numpy as np
import keras
import sklearn
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical as cg
from keras.callbacks import LearningRateScheduler
from sklearn import svm
from sklearn.metrics import accuracy_score

batch_size = 128

def unpickle(file):
    with open(file, 'rb') as lol:
        opened_file = PIQL.load(lol)
    return opened_file

def get_input_data():
    path = sys.argv[1]
    testpath = sys.argv[2]
    return path, testpath

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
    # y_test = cg(t['labels'], 10)
    scamx = a['data']
    scamy = cg(a['labels'],10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = (x_train - 80) / 255
    x_test = (x_test - 80) / 255
    tempy_train = np.concatenate((a['labels'], b['labels'], c['labels'], d['labels'], e['labels']), axis=0)
    tempy_test = t['labels']

    return label_names, x_train, y_train, x_test, scamx, scamy,tempy_train, tempy_test

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


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def final_pred(model,data):
    return model.predict(data)

def func_predict(inp_train,tempy_train,inp_test):
    clf = svm.LinearSVC(verbose=0, max_iter=1500, multi_class='ovr')
    clf.fit(inp_train, tempy_train)
    return clf.predict(inp_test)

def print_op(label_names,op):
    f = open('q2_c_output.txt', 'w')
    for i in op:
        f.write(label_names[i] + '\n')
    f.close()

if __name__ == '__main__':
    path , testpath = get_input_data()
    label_names, x_train, y_train, x_test, scamx, scamy, tempy_train, tempy_test = pre_process(path, testpath)
    model = build_model(x_train, y_train, x_test, y_test)
    x_train_size = x_train.shape[0]
    np.random.seed(1000)
    epochs = 15
    datagen = build_generator(x_train)

    for i in xrange(15):

        model.fit_generator(datagen.flow(x_train, y_train, batch_size=128, shuffle=True),
                  steps_per_epoch=x_train_size // batch_size,
                  epochs=1,
                  validation_data=(scamx, scamy),
                  workers=6, verbose=0)

    for i in xrange(7):
        pop_layer(model)

    model.build()

    print model.summary()

    inp_train = final_pred(model,x_train)
    inp_test = final_pred(model,x_test)

    preds = func_predict(inp_train,tempy_train,inp_test)

    for i in preds:
        print label_names[i]

    print accuracy_score(tempy_test, preds)

    op = final_pred(model,x_test)
    print_op(label_names,op)
