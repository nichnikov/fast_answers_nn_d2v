from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, Input, Conv1D, Flatten, Lambda
from keras.models import Model, Sequential
import numpy as np

from keras.optimizers import RMSprop
from keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(5000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1500, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(500, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# функция векторизации, возвращает список:
#  кортежей 
# if one_vector == True [(v: np.array, lb: int), ...]
# if one_vector == False [((v1: np.array, v2: np.array), lb: int), ...]

def siamese_data_prepare(data_tuples, d2v_model):
    X, y = [], []
    for q1, q2, lb in data_tuples:
        v1 = d2v_model.infer_vector(q1.split())
        v2 = d2v_model.infer_vector(q2.split())
        X.append((v1, v2))
        y.append(lb)
    return X, y


if __name__ == "__main__":
    model_path = r"./models"
    data_path = r"./data"
    model_name = "bss_d2v_nn_siamese_gr55_1021.h5"
    d2v_model = Doc2Vec.load(os.path.join(model_path, 'bss_doc2vec_model'))
    train_dataset = "dataset_train_gr55.csv"
    validate_dataset = "dataset_validate_gr55.csv"

    # загрузим сразу подготовленные для обучения данные
    train_data_df = pd.read_csv(os.path.join(data_path, train_dataset))
    train_data_df = train_data_df.sample(frac=1)
    train_data_tuples = zip(list(train_data_df["question1"]), list(train_data_df["question2"]),
                            list(train_data_df["is_duplicate"]))

    test_data_df = pd.read_csv(os.path.join(data_path, validate_dataset))
    test_data_df = test_data_df.sample(frac=1)
    test_data_tuples = zip(list(train_data_df["question1"]), list(train_data_df["question2"]),
                           list(train_data_df["is_duplicate"]))

    x_train, y_train = siamese_data_prepare(train_data_tuples, d2v_model)
    x_test, y_test = siamese_data_prepare(test_data_tuples, d2v_model)

    # ===========================================================================================
    epochs = 45

    tr_pairs = np.array(x_train)
    tr_y = np.array(y_train)

    te_pairs = np.array(x_test)
    te_y = np.array(y_test)

    input_shape = (300, 1)
    print(input_shape)

    # network definition
    base_network = create_base_network(input_shape)
    print("base_network Done")

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    print("processed_a, processed_b Done")

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    print("distance Done")

    model = Model([input_a, input_b], distance)
    print("model Done")

    # train
    model.compile(loss=contrastive_loss, optimizer="Adam", metrics=[accuracy])
    print("model.compile Done")
    print("tr_pairs.shape:", tr_pairs.shape)

    x_tr1 = tr_pairs[:, 0].reshape(tr_pairs[:, 0].shape[0], tr_pairs[:, 0].shape[1], 1)
    x_tr2 = tr_pairs[:, 1].reshape(tr_pairs[:, 1].shape[0], tr_pairs[:, 1].shape[1], 1)

    x_te1 = te_pairs[:, 0].reshape(te_pairs[:, 0].shape[0], te_pairs[:, 0].shape[1], 1)
    x_te2 = te_pairs[:, 1].reshape(te_pairs[:, 1].shape[0], te_pairs[:, 1].shape[1], 1)

    x_tr1 = x_tr1.astype(np.float32)
    x_tr2 = x_tr2.astype(np.float32)
    tr_y = tr_y.astype(np.float32)
    x_te1 = x_te1.astype(np.float32)
    x_te2 = x_te2.astype(np.float32)
    te_y = te_y.astype(np.float32)
    model.fit([x_tr1, x_tr2], tr_y,
              batch_size=512,
              epochs=epochs,
              validation_data=([x_te1, x_te2], te_y))

    model.save(os.path.join(model_path, model_name))
    print("x_tr1.shape:", x_tr1.shape)
    print("x_tr2.shape:", x_tr1.shape)
    print("tr_y.shape:", tr_y.shape)

    print("x_te1.shape:", x_te1.shape)
    print("x_te2.shape:", x_te1.shape)
    print("te_y.shape:", te_y.shape)