import config
from preprocess import get_modified_data
from DeepFM import DeepFM

import numpy as np
import pandas as pd
from time import perf_counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC


def get_data():
    file = pd.read_csv('../data/adult.data', header=None)

    X = file.loc[:, 0:13]
    Y = file.loc[:, 14].map({' <=50K': 0, ' >50K': 1})
    X.columns = config.ALL_FIELDS
    field_dict, field_index, X_modified = get_modified_data(X, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS, False)
    X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train.values, tf.float32), tf.cast(Y_train, tf.float32))) \
        .shuffle(30000).batch(config.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test.values, tf.float32), tf.cast(Y_test, tf.float32))) \
        .shuffle(10000).batch(config.BATCH_SIZE)
    print(test_ds)
    return train_ds, test_ds, field_dict, field_index


# Batch 단위 학습
def train_on_batch(model, optimizer, acc, auc, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=targets, y_pred=y_pred)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy & auc
    acc.update_state(targets, y_pred)
    auc.update_state(targets, y_pred)

    return loss


# 반복 학습 함수
def train(epochs):
    train_ds, test_ds, field_dict, field_index = get_data()

    model = DeepFM(embedding_size=config.EMBEDDING_SIZE, num_feature=len(field_index),
                   num_field=len(field_dict), field_index=field_index)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    print("Start Training: Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    start = perf_counter()
    for i in range(epochs):
        acc = BinaryAccuracy(threshold=0.5)
        auc = AUC()
        loss_history = []

        for x, y in train_ds:
            loss = train_on_batch(model, optimizer, acc, auc, x, y)
            loss_history.append(loss)

        print("Epoch {:03d}: 누적 Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}".format(
            i, np.mean(loss_history), acc.result().numpy(), auc.result().numpy()))

    test_acc = BinaryAccuracy(threshold=0.5)
    test_auc = AUC()
    for x, y in test_ds:
        print(x)
        y_pred = model(x)
        test_acc.update_state(y, y_pred)
        test_auc.update_state(y, y_pred)

    print("테스트 ACC: {:.4f}, AUC: {:.4f}".format(test_acc.result().numpy(), test_auc.result().numpy()))
    print("Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    print("걸린 시간: {:.3f}".format(perf_counter() - start))
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    model.save('../weights/weights-epoch({})-batch({})-embedding({})2.h5py'.format(
        epochs, config.BATCH_SIZE, config.EMBEDDING_SIZE))


if __name__ == '__main__':
    # train(epochs=1000)
    load_model = tf.keras.models.load_model('../weights/weights-epoch(1000)-batch(256)-embedding(5)2.h5py')

    file = pd.read_csv('../data/adult.data', header=None)
    X = file.loc[:, 0:13]
    X.columns = config.ALL_FIELDS

    field_dict, field_index, X_modified = get_modified_data(X, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS, False)
    # print(X_modified.columns)

    test_data = pd.DataFrame([[30, ' State-gov', ' 141297', ' Bachelors', 13, ' Married-civ-spouse', ' Prof-specialty', ' Husband', ' Asian-Pac-Islander', ' Male', 0, 0, 40, ' India']], columns=config.ALL_FIELDS)
    test_data = test_data.astype(dict(zip(config.ALL_FIELDS, config.DATA_TYPES)))
    X = X.append(test_data, ignore_index=True)
    _, _, test_data = get_modified_data(X, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS, False)
    test_data = pd.DataFrame(test_data.iloc[-1]).transpose()
    print(load_model.predict(test_data))
