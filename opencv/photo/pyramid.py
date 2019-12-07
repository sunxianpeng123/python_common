# encoding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_img(name="test",img=None):
    plt.figure()
    channel = img.shape
    if channel == 3:
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    path = r"lena.jpg"
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,127,255,0)
    """实现傅里叶变换"""
    # 对图像进行傅里叶变换
    dft = cv2.dft(np.float32(gray),flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将低频从左上角移动到中心
    dshift = np.fft.fftshift(dft)
    # 重置 区间,映射到[0,255]之间，以便使用图像显示
    # result = 20 * np.log(cv2.magnitude(dshift[:,:,0],dshift[:,:,1]))

    """低通滤波"""
    rows,cols = gray.shape
    # 取图像中心点
    row_mid,col_mid = int(rows / 2), int(cols / 2)
    # 构造掩膜图像
    mask = np.zeros((rows,cols,2),np.uint8)

    # 去掉低频区域
    mask[row_mid-30:row_mid+30,col_mid-30:col_mid+30] = 1
    # 将频谱图像和掩膜相乘,保留低频部分，高频部分变成0
    md = dshift * mask

    """实现逆傅里叶变换"""
    # 将低频从中心移动到左上角
    imd = np.fft.ifftshift(md)
    # 返回一个复数数组
    iimg = cv2.idft(imd)
    # 将上述复数数组重置区间到[0,255],便于图像显示
    iimg = cv2.magnitude(iimg[:,:,0],iimg[:,:,1])

    # 原始灰度图
    plt.subplot(1,2,1 ),plt.imshow(gray,'gray'),plt.axis('off'),plt.title('original')
    # 频谱图像
    plt.subplot(1,2,2),plt.imshow(iimg,'gray'),plt.axis('off'),plt.title('result')
    plt.show()