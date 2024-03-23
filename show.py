import keras
import numpy as np
import matplotlib.pyplot as plt
from grad_cam import grad_cam

Data = np.load('D:/acc_model/data1.npy')
Data = np.array(Data)
labels = np.load('D:/acc_model/label1.npy')
print(labels.shape)
model = keras.models.load_model('D:/acc_model/T-model2.hdf5')#my_model
indexs = np.load('D:/acc_model/view/pred-n.npy')
# indexs = np.load('D:/acc_model/view/pred-af.npy')

def Normalization(x, min, max):
    x = (x - min) / (max - min)
    return x

for i in range(10, 150000):

    index = int(indexs[i])
    label = int(labels[i])
    # label = int(labels[i + 150000]) # af

    if index == 0 and label == 0:
        print(index)
        # datas = Data[i+150000, :]
        datas = Data[i, :]
        data = np.expand_dims(datas, 0)
        data = np.expand_dims(data, 2)
        heatmap = grad_cam(model, data, category_index=1, layer_name='conv2d_6', nb_classes=2) #改网络名字

        fig, ax1 = plt.subplots()

        ax1.plot(datas, color='blue', linestyle='-')
        ax2 = ax1.twinx()
        # w = np.mean(heatmap)
        heatmap.copy()
        heatmap = Normalization(heatmap, min(heatmap), max(heatmap))
        # heatmap[heatmap < 0.15] = 0
        # ax2.plot(heatmap, 'r--')
        heatmap_x = list(range(1, 1001))
        plt.fill_between(heatmap_x, heatmap, 0, color='r', alpha=0.4)

        # plt.savefig('Ischemic_heart_disease'+str(i)+'.jpg', bbox_inches='tight', dpi=1080)
        plt.show()


