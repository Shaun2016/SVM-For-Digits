from sklearn.externals import joblib
from ImageFactory import ImageToVector

if __name__ == '__main__':
    clf = joblib.load('train_model.m')  # 载入 svm 模型
    scalar = joblib.load('scalar')  # 载入 归一化 模型
    pic_name = input()  # 输入图片名
    pic = ImageToVector(pic_name)   # 将图片转变为向量
    pic = scalar.transform(pic)     # 归一化向量
    y_predict = clf.predict(pic)    # 使用模型预测
    print(y_predict)                # 显示预测结果
