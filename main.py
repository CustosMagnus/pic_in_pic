import time

import cv2
import os
import numpy as np

resize = 25
y_s_img = 150
x_s_img = 100

def r_main():
    print(f"reading {os.listdir('./main_picture')[0]} ...")
    o = cv2.imread(f"./main_picture/{os.listdir('./main_picture')[0]}")
    return o

def all_pics():
    return cv2.imread(f"./main_picture/{os.listdir('./pictures')}")

def r_img(img):
    return cv2.imread(img)

def check_img(name):
    img = cv2.imread(f"./pictures/{name}")
    l = os.listdir("./pictures/calc")
    if name in l:
        return None
    if img is None:
        return None
    elif 0.6 < (img.shape[1]/img.shape[0]) < 0.70:
        return img
    elif img.shape[1]/img.shape[0] > 1: # dann horizontal
        img2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        dim = img2.shape
        v = dim[1] * (2 / 3)
        x = (dim[0] - v) / 2
        x_s = v + x
    else:
        dim = img.shape
        v = dim[1] * (2 / 3)
        x = (dim[0] - v) / 2
        x_s = v + x
    return img[0:-1, int(round(x)):int(round(x_s))]

def brightness(img): # value between 0 and 255
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    br = np.mean(g_img)
    return br

def get_p_list():
    plist_o = os.listdir('./pictures')
    for i in range(len(plist_o)):
        img = check_img(plist_o[i])
        if img is not None:
            cv2.imwrite(f"./pictures/calc/{plist_o[i]}", img)
            print(f"scaling pictures: {(i+1)}/{len(plist_o)} ({plist_o[i]})")
    plist_o = os.listdir('./pictures/calc')
    return plist_o

def sort_p_list(plist):
    plist_val = np.zeros(len(plist), dtype=int)
    plist_val2 = np.zeros(len(plist), dtype=int)
    for i in range(len(plist)):
        rea = r_img(f"./pictures/calc/{plist[i]}")
        plist_val[i] = brightness(rea)
        print(f"calculating mean brightness: {i+1}/{(len(plist))}")
    for i in range(len(plist_val)):
        plist_val2[i] = plist_val[i]
    print("sorting pictures...")
    for i in range(len(plist)):
        pos = np.argmin(plist_val)
        s = plist[i]; plist[i] = plist[pos]; plist[pos] = s
        plist_val[pos] = 1000
        s = plist_val[i]; plist_val[i] = plist_val[pos]; plist_val[pos] = s
        s = plist_val2[i]; plist_val2[i] = plist_val2[pos]; plist_val2[pos] = s
    return [plist, plist_val2]

def fill(img, plist, oimg):
    oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2GRAY)
    step = 255/len(plist)
    s_y = oimg.shape[0]
    s_x = oimg.shape[1]
    s_t = s_y * s_x
    for fy in range(s_y):
        for fx in range(s_x):
            # choose picture
            br = oimg[fy][fx]
            img_n = cv2.imread(f"./pictures/calc/{plist[int(round(br/step))]}")
            img_n = cv2.resize(img_n, (x_s_img, y_s_img))
            for y in range(y_s_img):
                for x in range(x_s_img):
                    # fill pixel
                    for d in range(3):
                        img[fy*y_s_img+y][fx*x_s_img+x][d] = img_n[y][x][d]
            print(f"rendering: {fy*s_x+fx+1}/{s_t}")
    return img

def run():
    img = r_main()
    dim = img.shape
    print("dimensions: ", dim)
    img = cv2.resize(img, (int(round((dim[0]*(2/3)))), int(round(dim[1]))))
    dim = img.shape
    img = cv2.resize(img, (int(round(dim[0]/resize)), int(round(dim[1]/resize))))
    plist = get_p_list()
    re = sort_p_list(plist) # sorted from bright to dark
    plist = re[0]
    plist_val = re[1]
    nimg = np.zeros([int(round((dim[0] / resize) * y_s_img, 0)), int(round((dim[1] / resize) * x_s_img, 0)), int(dim[2])], dtype=np.uint8)
    print("array size: ", (nimg.nbytes / (1024 * 1024)), "Mb\ndimensions: ", nimg.shape)
    nimg = fill(nimg, plist, img)
    l = os.listdir("./output")
    if l != []:
        cv2.imwrite(f"./output/{int(l[-1].split('.')[0])+1}.jpg", nimg)
    else:
        cv2.imwrite("./output/1.jpg", nimg)
    cv2.imshow('', nimg)
    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    run()


