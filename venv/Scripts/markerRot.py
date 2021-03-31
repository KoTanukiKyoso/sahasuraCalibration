import cv2
import cv2.aruco as aruco
import numpy as np
from math import pi

arucoMarkerLength = 0.05


def out_translate_txt(txt_path, Rx, Ry, Rz, Tx, Ty, Tz):
    with open(txt_path, mode='w') as f:
        f.write(str(Rx)+ "," + str(Ry) + "," + str(Rz) + "\n" + str(Tx)+ "," + str(Ty) + "," + str(Tz))

class AR():

    def __init__(self, videoPort, cameraMatrix, distortionCoefficients):
        self.cap = cv2.VideoCapture(videoPort)
        self.videoPort=videoPort
        # self.cameraMatrix = np.load(cameraMatrix)
        # self.distortionCoefficients = np.load(distortionCoefficients)
        self.cameraMatrix = cameraMatrix
        self.distortionCoefficients = distortionCoefficients
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    def find_ARMarker(self):
        self.ret, self.frame = self.cap.read()
        if len(self.frame.shape) == 3:
            self.Height, self.Width, self.channels = self.frame.shape[:3]
        else:
            self.Height, self.Width = self.frame.shape[:2]
            self.channels = 1
        self.halfHeight = int(self.Height / 2)
        self.halfWidth = int(self.Width / 2)
        self.corners, self.ids, self.rejectedImgPoints = aruco.detectMarkers(self.frame, self.dictionary)
        # corners[id0,1,2...][][corner0,1,2,3][x,y]
        aruco.drawDetectedMarkers(self.frame, self.corners, self.ids, (0, 255, 0))

    def show(self,videoport):
        cv2.imshow(str(videoport)+"result", self.frame)
    def save(self,videoport):
        cv2.imwrite("img"+str(videoport)+".jpg",self.frame)

    def get_exist_Marker(self):
        return len(self.corners)

    def is_exist_marker(self, i):
        num = self.get_exist_Marker()
        if i >= num:
            return False
        else:
            return True

    def release(self):
        self.cap.release()

    # マーカー頂点の座標を取得
    def get_ARMarker_points(self, i):
        if self.is_exist_marker(i):
            return self.corners[i]

    def get_average_point_marker(self, i):
        if self.is_exist_marker(i):
            points = self.get_ARMarker_points(i)
            points_reshape = np.reshape(np.array(points), (4, -1))
            G = np.mean(points_reshape, axis=0)
            cv2.circle(self.frame, (int(G[0]), int(G[1])), 10, (255, 255, 255), 5)
            return G[0], G[1]

    def get_ARMarker_pose(self, i):
        if self.is_exist_marker(i):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(self.corners[i], arucoMarkerLength, self.cameraMatrix,
                                                            self.distortionCoefficients)
            self.frame = aruco.drawAxis(self.frame, self.cameraMatrix, self.distortionCoefficients, rvec, tvec, 0.1)
            return rvec, tvec


    def get_degrees(self, i):
        if self.is_exist_marker(i):
            rvec, tvec, = self.get_ARMarker_pose(i)
            print(str(rvec)+":"+str(tvec))
            filepath="Translate_"+str(self.videoPort)+".txt"
            out_translate_txt(filepath, rvec[0][0][0], rvec[0][0][1], rvec[0][0][2],tvec[0][0][0], tvec[0][0][1], tvec[0][0][2])

            (roll_angle, pitch_angle, yaw_angle) = rvec[0][0][0] * 180 / pi, rvec[0][0][1] * 180 / pi, rvec[0][0][
                2] * 180 / pi
            if pitch_angle < 0:
                roll_angle, pitch_angle, yaw_angle = -roll_angle, -pitch_angle, -yaw_angle
            return roll_angle, pitch_angle, yaw_angle


if __name__ == '__main__':

    camera_matrix = np.matrix([[627.8095, 0.0, 325.2734], [0.0,630.1833, 197.4435], [0.0, 0.0, 1.0]])
    distortion = np.array([0.02524192, -0.15412883, -0.00789069, -0.00260461, 0.04049524])
    #camera_matrix = np.matrix([[332.85998679,0.,226.17921689],[  0. ,322.02029255,238.23030038], [0.0, 0.0, 1.0]])
    #distortion = np.array([-0.07458388  ,0.03687577 ,-0.00575752 , 0.00042157,  0.04391123])

    i = 0
    flag = True
    captures = []

    while (flag):
        capture = cv2.VideoCapture(i)
        ret, frame = capture.read()
        flag = ret
        if flag:
            captures.append(AR(i,camera_matrix, distortion))
            i += 1


    #myCap = AR(0, camera_matrix, distortion)
    #myCap2=AR(1, camera_matrix, distortion)
    while True:

        for i, myCap in enumerate(captures):
            myCap.find_ARMarker()
            myCap.get_average_point_marker(0)
            myCap.show(i)
        if cv2.waitKey(1) > 0:

            for i, myCap in enumerate(captures):
                print(myCap.get_degrees(0))
                myCap.save(i)
                myCap.release()

                cv2.destroyAllWindows()
            break

"""
        myCap.find_ARMarker()
        myCap.get_average_point_marker(0)
        #print(myCap.get_degrees(0))
        myCap.show(0)
        myCap2.find_ARMarker()
        myCap2.get_average_point_marker(0)
        #print(myCap2.get_degrees(0))
        myCap2.show(1)


        if cv2.waitKey(1) > 0:

            print(myCap.get_degrees(0))
            print(myCap2.get_degrees(0))
            myCap.save(0)
            myCap2.save(1)
            myCap.release()
            myCap2.release()
            cv2.destroyAllWindows()
            break
"""