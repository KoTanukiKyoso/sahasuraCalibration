import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def arGenerator():
    fileName = "ar.png"
    generator = aruco.drawMarker(dictionary, 3, 100)
    cv2.imwrite(fileName, generator)

    img = cv2.imread(fileName)
    cv2.imshow('ArMarker',img)
    cv2.waitKey(0)
arGenerator()
