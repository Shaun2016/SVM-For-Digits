from PIL import Image
import numpy as np
import os

"""
将一个图片转为数值向量
"""
def ImageToVector(pic, length=64):
    im = Image.open(pic)
    data = np.asarray(im)
    data = np.reshape(data, (1, length))
    return data.astype(float)

"""
将指定路径下所有的图片转为数值向量并存入 CSV 文件中
"""
def ImageToCSV(pic_path):
    pics = os.listdir(pic_path)
    matrix = []
    for picture in pics:
        v = ImageToVector(pic_path + '/' + picture)
        category = 1    # 图片的类别
        v = np.append(v[0], category).astype(int)
        matrix.append(v)
    np.savetxt('data/new.csv', matrix, delimiter=',')

if __name__ == '__main__':
    ImageToCSV('images')
