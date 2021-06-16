import sys
import cv2 as cv
import numpy as np

def preProcesamiento(img):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)    
    imgCanny = cv.Canny(imgBlur,10,200)
    kernel = np.ones((2,2))
    imgDial = cv.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv.erode(imgDial,kernel,iterations=1)    
    return imgThres

def getContornoBillete(img):
    imgContornos = img.copy()
    imgEdges = preProcesamiento(img)
    contornos, jerarquia = cv.findContours(imgEdges,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) 
    areaMasGrande = 0
    contornoMasGrande = 0
    for contorno in contornos:
        area = cv.contourArea(contorno) 
        if area > areaMasGrande:
            areaMasGrande = area
            contornoMasGrande = contorno

    #cv.drawContours(imgContornos,contornoMasGrande,-1,(0,255,0),3)
    perimetro = cv.arcLength(contornoMasGrande,True)                
    #aproximar esquinas
    approx = cv.approxPolyDP(contornoMasGrande,0.02*perimetro, True)

    x, y, w, h = cv.boundingRect(approx)
    #cv.rectangle(imgContornos,(x,y),(x+w,y+h),(255,0,0),2)                
    alto = h + y
    largo = w + x

    retVal = imgContornos[y:alto,x:largo]
    return retVal


def main(argv):
    foto = cv.imread(argv[0])
    img = getContornoBillete(foto)
    cv.imshow("IMG",img)
    cv.waitKey(0)    


if __name__ == "__main__":
    main(sys.argv[1:])