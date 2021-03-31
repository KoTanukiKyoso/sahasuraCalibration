import cv2
import time
import argparse
import concurrent.futures
import glob
import numpy as np
import matplotlib.pyplot as plt


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def out_param_txt(txt_path, Fx,Fy,Cx,Cy):
    with open(txt_path, mode='w') as f:
        f.write(str(Fx)+","+str(Fy)+"\n"+str(Cx)+","+str(Cy))

def out_translate_txt(txt_path, Rx,Ry,Rz,Tx,Ty,Tz):
    with open(txt_path, mode='w') as f:
        f.write(str(Rx)[1:-1]+","+str(Ry)[1:-1]+","+str(Rz)[1:-1]+"\n"+str(Tx)[1:-1]+","+str(Ty)[1:-1]+","+str(Tz)[1:-1])



camera0 = cv2.VideoCapture(0)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

Ptxt_path="./Params.txt"#内部パラメータ（カメラ行列）
Ttxt_path="./Translate.txt"
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)



# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('../../obj_img/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    # If found, add object points, image points (after refining them)
    if ret== True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
   # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)





while(True):
    # フレームをキャプチャする
    ret0, frame0 = camera0.read()

    # 画面に表示する
    cv2.imshow('frame0', frame0)

    gray = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret== True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        frame0 = cv2.drawChessboardCorners(frame0, (7,6), corners2,ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        cv2.imshow('img1',frame0)
        cv2.waitKey(500)
        print(ret)  # 最小化問題を解いた際の最終的な再投影誤差のRMS(Root Mean Square)
        print(mtx)  # カメラ行列
        print(dist)  # 歪みパラメータのリスト
        print(rvecs)  # モデル座標系からカメラ座標系への回転ベクトル
        print(tvecs)  # モデル座標系からカメラ座標系への並進ベクトル
        out_param_txt(Ptxt_path, mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2])
        out_translate_txt(Ttxt_path, rvecs[len(rvecs)-1][0], rvecs[len(rvecs)-1][1], rvecs[len(rvecs)-1][2], tvecs[len(tvecs)-1][0], tvecs[len(tvecs)-1][1], tvecs[len(tvecs)-1][2])

    # キーボード入力待ち
    key = cv2.waitKey(1) & 0xFF

    # qが押された場合は終了する
    if key == ord('q'):
            break

camera0.release()
cv2.destroyAllWindows()