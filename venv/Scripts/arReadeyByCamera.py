#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def arReader():
    cap = cv2.VideoCapture(0) #ビデオキャプチャの開始

    while True:

        ret, frame = cap.read() #ビデオキャプチャから画像を取得

        Height, Width = frame.shape[:2] #sizeを取得

        img = cv2.resize(frame,(int(Width),int(Height)))

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary) #マーカを検出

        aruco.drawDetectedMarkers(img, corners, ids, (0,255,0)) #検出したマーカに描画する

        cv2.imshow('drawDetectedMarkers', img) #マーカが描画された画像を表示

        cv2.waitKey(1) #キーボード入力の受付

    cap.release() #ビデオキャプチャのメモリ解放
    cv2.destroyAllWindows() #すべてのウィンドウを閉じる


arReader()