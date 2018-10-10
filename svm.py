from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    # 数据预处理的工具类
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import numpy as np
from sklearn.datasets.base import Bunch
import os
from ntpath import join

class My_SVM(object):

    def load_file(self, filename, width=8, height=8):
        module_path = os.path.dirname(__file__)
        data = np.loadtxt(join(module_path, 'data', 'digits.csv'),
                          delimiter=',')    # 读取图数值向量文件
        target = data[:, -1].astype(np.int)  # 每行最后一个数字为目标分类 target
        flat_data = data[:, :-1]    # 每行除了最后一个数字是图片向量
        images = flat_data.view()   # 向量转图片
        images.shape = (-1, width, height)  # 变维: 长*宽

        return Bunch(data=flat_data,
                     target=target,
                     target_names=np.arange(10),
                     images=images)

    def svn_model_build(self):
        digits = self.load_file('data/digits.csv')

        print(digits.data.shape)
        # 将数据集分为训练集 和 测试集
        x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                            test_size=0.25, random_state=33)
        ss = StandardScaler()  # 创建标准化(归一化)对象
        x_train = ss.fit_transform(x_train)  # 分两步：1. fit(计算需要的参数：mean, std) 2. transform(归一化训练数据)
        x_test = ss.transform(x_test)  # 由于已经计算出了需要的参数 mean 和 std，只需要做归一化
        lsvc = SVC()    # 创建 SVM
        lsvc.fit(x_train, y_train)  # 训练集训练模型
        joblib.dump(lsvc, 'train_model.m')  # 保存训练好的模型结果
        joblib.dump(ss, 'scalar')  # 保存归一化模型
        y_predict = lsvc.predict(x_test)  # 测试机测试模型
        print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))   # 输出模型测试结果参数


if __name__ == '__main__':
    My_SVM().svn_model_build()
