from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

FEATURES = 57
np.random.seed(1)
file = 'spam.txt'
data = np.loadtxt(file, delimiter=' ', dtype=float)

np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

Xm = np.zeros(len(y))
Xs = np.zeros(len(y))
for i in range(np.shape(X)[1]):
    Xm = np.mean(X[:, i])
    Xs = np.std(X[:, i], ddof=1)
    X[:, i] = (X[:, i]-Xm)/Xs
'''
for j in range(len(y)):
    if y[j] == 0:
        y[j] = -1
'''
indices = np.random.permutation(X.shape[0])
num = X.shape[0] // 5

rg = indices[0:num]
X_test = X[rg][:]
y_test = y[rg]

X_train = np.delete(X, rg, axis=0)
y_train = np.delete(y, rg, axis=0)
# analysis of number of hidden units.
'''for i in range(10):
    unum = 2**(i+1)
    model = keras.Sequential([
    keras.layers.Dense(unum, activation='sigmoid', use_bias='FALSE', input_shape=(FEATURES,)),
    keras.layers.Dense(1, activation='sigmoid', use_bias='FALSE')
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    his = model.fit(X_train, y_train, epochs = 100, validation_split = 0.5, verbose=2)

    history_dict = his.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    b1 = val_loss.index(min(val_loss))+1
    epochs = range(1, len(acc) + 1)

    plt.plot(b1, val_loss[b1-1], 'ro')
    # “bo”代表 "蓝点"
    plt.plot(epochs, loss, 'b', label='Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    print('\n', b1)
    print('\n', val_loss[b1-1])
    plt.show()
'''
for i in range(5):
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation='sigmoid', use_bias='FALSE', input_shape=(FEATURES,)))
    for j in range(i):
        model.add(keras.layers.Dense(10, activation='sigmoid', use_bias='FALSE'))
    # 添加一个全连接层，1个输入，10个输出，激活函数为tanh
    model.add(keras.layers.Dense(1, activation='sigmoid', use_bias='FALSE'))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    his = model.fit(X_train, y_train, epochs=100, validation_split=0.5, verbose=2)

    history_dict = his.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    b1 = val_loss.index(min(val_loss)) + 1
    epochs = range(1, len(acc) + 1)
    plt.plot(b1, val_loss[b1 - 1], 'ro')
    # “bo”代表 "蓝点"
    plt.plot(epochs, loss, 'b', label='Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    print('\n', b1)
    print('\n', val_loss[b1 - 1])
    plt.show()