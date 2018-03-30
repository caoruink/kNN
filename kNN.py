# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class DataSet(object):
    """
    1.读取文件中的数据集，
    2.按照需要拆成训练集、验证集、测试集，
    3.使用最近邻决策方法(kNN)分类
    """
    def __init__(self, filename):
        """
        读取数据
        :param filename: 读取数据的文件名称
        """
        self.__data_set = np.loadtxt(filename, delimiter=',')
        self.X = self.__data_set[:, 0: 4]       # 特征
        self.Y = self.__data_set[:, 4]          # 目标
        self.__num_split = 10                   # 交叉验证的数目

    def classify(self):
        """
        1.分别使用参数1~120 的kNN 分类器分类iris 数据集
        2.使用交叉验证方法确定最优k值
        :return:
        """
        # 使用参数k=1~120训练kNN,记录分类错误率
        print("使用参数k=1~120训练kNN,记录分类错误率\n")
        kf = KFold(n_splits=self.__num_split, shuffle=True)
        # 区分训练集和测试集
        all_error = []
        for train_index, test_index in kf.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            all_error.append(self.kNN_test(x_train, x_test, y_train, y_test))
        all_error = np.mean(all_error, axis=0)
        print("画出分类错误率曲线\n\n")
        plt.plot(np.arange(1, 121), all_error)
        plt.ylabel('error_rate')
        plt.xlabel('k')
        plt.title('The error rate of various k in kNN')
        plt.savefig("error_rate.jpg")
        plt.show()

        # 交叉验证
        for train_index, test_index in kf.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            self.cross_validate(x_train, x_test, y_train, y_test)

    @staticmethod
    def __equal(x, y):
        """
        判断利用分类规则决策的结果于原结果是否一样
        :param x: 决策的类别
        :param y: 原结果
        :return: 一样，返回真，否则返回假
        """
        if x == y:
            return True
        return False

    def kNN_test(self, x_train, x_test, y_train, y_test):
        """
        kNN训练，不做验证，k=1~120
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        ks = np.arange(1, 121)
        error = []
        for k in ks:
            error.append(self.nearest_neighbor_decision(k, x_train, x_test, y_train, y_test))
        return error

    def cross_validate(self, x_train, x_test, y_train, y_test):
        """
        交叉验证，选择最优的k
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        print("交叉验证，选取最优的k\n")
        ks = np.arange(1, 40)
        error_all = []
        kf_val = KFold(n_splits=self.__num_split - 1, shuffle=True)
        # 把训练集再拆成训练集和验证集，求最优的k
        for train_new_index, val_index in kf_val.split(x_train):
            x_train_new, x_val = x_train[train_new_index], x_train[val_index]
            y_train_new, y_val = y_train[train_new_index], y_train[val_index]
            error = []
            for k in ks:
                error.append(self.nearest_neighbor_decision(k, x_train_new, x_val, y_train_new, y_val))
            error_all.append(error)
        error_all = np.mean(error_all, axis=0).tolist()
        best_k = error_all.index(min(error_all))
        print("最优的参数k是", str(best_k + 1), '\n')
        print("以最优参数进行训练和测试\n")
        error = []
        for train_new_index, val_index in kf_val.split(x_train):
            x_train_new = x_train[train_new_index]
            y_train_new = y_train[train_new_index]
            error.append(self.nearest_neighbor_decision(best_k + 1, x_train_new, x_test, y_train_new, y_test))
        print(error)
        print("平均错误分类率为" + str(mean(error)), '\n')
        print("最大错误分类率为" + str(max(error)), '\n')
        print("最小错误分类率为" + str(min(error)), '\n')
        print("错误分类率方差为" + str(var(error)), '\n')
        return

    def nearest_neighbor_decision(self, k, x_train, x_test, y_train, y_test):
        """
        根据训练集确定距离，利用最近邻决策确定结果
        :param k: 参数
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        num_error = 0
        for index_test in np.arange(0, len(x_test)):
            prob = self.find_k_nearest_point(k, x_train, y_train, x_test[index_test])
            if max(prob) == prob[0]:
                decide_class = 1
            elif max(prob) == prob[1]:
                decide_class = 2
            else:
                decide_class = 3
            if not self.__equal(decide_class, y_test[index_test]):
                num_error += 1
        # 求平均的分类正确率
        # print("the error rate of the nearest_neighbor_decision is", num_error / len(x_test))
        return num_error / len(x_test)

    def find_k_nearest_point(self, k, x_train, y_train, x_test):
        # 计算k个与样本点最近的距离，记录标签
        distance = []
        for index_sam_train in np.arange(0, len(x_train)):
            temp_dis = {'dis': 0, 'label': 1}
            temp_dis['dis'] = self.__cal_distance(x_train[index_sam_train], x_test)
            temp_dis['label'] = y_train[index_sam_train]
            distance.append(temp_dis)
        distance = sorted(distance, key=lambda x: x['dis'])
        found_label = distance[0:k]
        # 分别记录每个标签的个数
        num_label1 = 0
        num_label2 = 0
        num_label3 = 0
        for each in found_label:
            if each['label'] == 1:
                num_label1 += 1
            elif each['label'] == 2:
                num_label2 += 1
            else:
                num_label3 += 1
        return [num_label1/k, num_label2/k, num_label3/k]

    @staticmethod
    def __cal_distance(x, y):
        """
        计算向量之间的欧氏距离
        :param x:四维矢量
        :param y:四维矢量
        :return:欧氏距离
        """
        return np.sqrt(np.square(x[0] - y[0]) + np.square(x[1] - y[1]) + np.square(x[2] - y[2]) + np.square(x[3] - y[3]))


if __name__ == '__main__':
    data_set = DataSet("iris.txt")
    data_set.classify()
