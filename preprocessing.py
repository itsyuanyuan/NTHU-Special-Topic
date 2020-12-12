import cv2
import os


path_ = 'C:\\Users\\aspy1\\Desktop\\img\\origin\\2'
try:
    os.mkdir(path_[:-1]+'3')
except:
    pass
c = 0
for i in os.listdir(path_):
    name = path_ + '\\' + i
    img = cv2.imread(name)
    img = cv2.resize(img,(256,256),interpolation = cv2.INTER_LANCZOS4)
    cv2.imwrite(path_[:-1]+'3' + "\\" + str(c) + '.jpg',img)
    c+=1
    cols = 256
    rows = 256
    img_flip = cv2.flip(img,1)
    cv2.imwrite(path_[:-1]+'3' + "\\" + str(c) + '.jpg',img_flip)
    c+=1
    for i in range(45,360,45):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
        img1 = cv2.warpAffine(img,M,(256,256))
        img2 = cv2.warpAffine(img_flip,M,(256,256))
        cv2.imwrite(path_[:-1]+'3' + "\\" + str(c) + '.jpg',img1)
        c+=1
        cv2.imwrite(path_[:-1]+'3' + "\\" + str(c) + '.jpg',img2)
        c+=1
