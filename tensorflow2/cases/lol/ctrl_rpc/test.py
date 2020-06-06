import cv2
import os
import numpy as np

print(cv2.__version__)


def getKillOrDead(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny边缘处理
    edges = cv2.Canny(gray, 50, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 8, minLineLength=8, maxLineGap=1)
    print("l", lines)
    if lines is None:
        return False
    else:
        return True

def getActionInfo(img):
    w = img.shape[1]
    m_w = int(w / 2)
    img_left = img[:, 0:m_w]
    img_right = img[:, m_w:w]
    result = {}
    result['kill'] = getKillOrDead(img_left)
    result['dead'] = getKillOrDead(img_right)
    return result


def findWhiteNum(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([15, 30, 160])
    # upper_red = np.array([70, 170, 250])
    # lower_red = np.array([20, 30, 160])
    # upper_red = np.array([70, 170, 250])
    lower_red = np.array([15, 35, 160])
    upper_red = np.array([50, 140, 250])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


img_path = "val/" + "6ddf9e74-774a-40ab-a484-4a0c9e1c17eetest.jpg"
import os
print("img path = {}".format(os.path.abspath(img_path)))
img = cv2.imread(img_path)
print("img.shape = {}".format(img.shape))
img1 = findWhiteNum(img)
result = getActionInfo(img1)
print(result)
