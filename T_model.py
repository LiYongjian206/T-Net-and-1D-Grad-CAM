from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.model_selection import train_test_split  # 划分数据集
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import optimizers
from sklearn import metrics  # 模型评估
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Add, Multiply, \
    GlobalAveragePooling1D, Concatenate, BatchNormalization, ELU, Conv2D, \
    Reshape, MaxPooling2D, AveragePooling2D, AveragePooling1D, GlobalMaxPooling1D
from keras import backend as K
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras_flops import get_flops

def shuffle_set(data, label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data, Label

# 学习率更新以及调整
def scheduler(epoch):
    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)  # keras默认0.001
        K.set_value(model.optimizer.lr, lr*10)
        print("lr changed to {}".format(lr))
    if epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        # K.set_value(model.optimizer.lr, lr * math.pow(0.99, epoch))
        K.set_value(model.optimizer.lr, lr / (1 + 0.0001 * epoch))
        print("lr changed to {}".format(lr))
    return K.get_value(model.optimizer.lr)

F1 = []
Con_Matr = []
# 数据导入
data = np.load('D:/acc_model/data_mit.npy')
label = np.load('D:/acc_model/label_mit.npy')
# data = np.load('D:/acc_model/data2.npy')
# label = np.load('D:/acc_model/label2.npy')
label = np_utils.to_categorical(label, 2)
Data, Label = shuffle_set(data, label)
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.7, random_state=32)

# # 10折
# data = np.load('D:\MEMS_CNN\Kf\data_k.npy')
# label = np.load('D:\MEMS_CNN\Kf\label_k.npy')
# label = np_utils.to_categorical(label, 2)
# Data, Label = shuffle_set(data, label)
# X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=1/(10**10), random_state=32)

D = 64
S = 128
F = 16

def Block1(input, k, c):

    conv1_1 = Conv1D(filters=c, kernel_size=k, strides=1)(input)
    conv1_1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_1)
    conv1_1 = ELU()(conv1_1)
    conv1_1 = MaxPooling1D(pool_size=2, strides=2)(conv1_1)
    return conv1_1

# 投票与特征筛选结合
def Filter(inputs, c):

    # input1 = GlobalAveragePooling1D()(inputs)
    # input2 = GlobalMaxPooling1D()(inputs)
    # input  = Add()([input1, input2])
    input = Flatten()(inputs)

    x = Dense(c/16, activation='relu')(input)
    x = Dense(c, activation='sigmoid')(x)

    y = Dense(c/16, activation='relu')(input)
    y = Dense(c, activation='sigmoid')(y)

    z = Dense(c/16, activation='relu')(input)
    z = Dense(c, activation='sigmoid')(z)

    out = Add()([x, y, z])

    return out/3

def models(input_shape):

    #Part_1
    # 1
    x1 = Block1(input_shape, k=3, c=D*2)
    y1 = Filter(x1, c=S)

    # 2
    x2 = Block1(x1, k=3, c=D*2)
    y2 = Filter(x2, c=S)

    # 3
    x3 = Block1(x2, k=3, c=D*2)
    y3 = Filter(x3, c=S)

    # 4
    x4 = Block1(x3, k=3, c=D*2)
    y4 = Filter(x4, c=S)

    # 5
    x5 = Block1(x4, k=3, c=D*2)
    y5 = Filter(x5, c=S)

    # 6
    x6 = Block1(x5, k=3, c=D*2)
    y6 = Filter(x6, c=S)

    # 7
    x7 = Block1(x6, k=3, c=D*2)
    y7 = Filter(x7, c=S)

    # 8
    x8 = Block1(x7, k=3, c=D*2)
    y8 = Filter(x8, c=S)

    # Part_2
    c1 = Concatenate()([y1, y2, y3, y4, y5, y6, y7, y8])
    c1 = Reshape((8, S, 1), input_shape=(None, 8 * S))(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)

    # 1
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # 2
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # 3
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # 4
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # 5
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # 6
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # 7
    c1 = Conv2D(filters=F, kernel_size=(2, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)

    # Out
    out = Flatten()(c1)
    out = Dense(2, activation='softmax')(out)
    out = Model(inputs=[input_shape], outputs=[out], name="T_CNN")
    return out

inputs = Input(shape=(1000,1))

model = models(inputs)
model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6:.05} M")

# sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

model.compile(loss='binary_crossentropy',
              optimizer='Adam', metrics='accuracy')

filepath = "D:/acc_model/mit.hdf5"  # 保存模型的路径

checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
                             monitor='val_accuracy', mode='max')
# , save_best_only='True'
reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变
callback_lists = [checkpoint, reduce_lr]

train_history = model.fit(x=X_train,
                          y=y_train, validation_split=0.1, verbose=2,
                          class_weight=None, callbacks=callback_lists,
                          epochs=50, batch_size=128)

loss, accuracy = model.evaluate(X_test, y_test)  # 修改损失函数

Acc = []
Loss = []
Acc.append(accuracy)
Loss.append(loss)

y_pred = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
f1 = metrics.f1_score(y_test, y_pred, average='macro')
F1.append(f1)
con_matr = confusion_matrix(y_test, y_pred)
Con_Matr.append(con_matr)
print(Con_Matr)
print(F1)

'''
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')  # 绘制准确率执行曲线
show_train_history(train_history, 'loss', 'val_loss')  # 绘制损失函数执行曲线

from sklearn.metrics import roc_curve#画roc曲线
from sklearn.metrics import auc#auc值计算

y_pred_keras = y_pred
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'r-.')
plt.plot(fpr_keras, tpr_keras,'--', label='newNet (area = {:.4f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()
'''
