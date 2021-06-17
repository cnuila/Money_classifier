import sys
import cv2 as cv
import numpy as np

def autoCanny(img):
    medianaVector = np.median(img)
    sigma = 0.33
    lowerThre = int(max(0,(1.0 - sigma) * medianaVector))
    upperThre = int(min(255,(1.0 - sigma) * medianaVector))
    imgCanny = cv.Canny(img,lowerThre,upperThre)
    return imgCanny

#0 wide, 1 tight, 2 automatico con mediana
def preProcesamiento(img,numCanny):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(3,3),1) 
   
    if numCanny == 0:
        imgCanny = cv.Canny(imgBlur,10,200)
    elif numCanny == 1:
        imgCanny = cv.Canny(imgBlur,255,250)
    else:
        imgCanny = autoCanny(imgBlur)

    kernel = np.ones((2,2))
    imgDial = cv.dilate(imgCanny,kernel,iterations=2)
    imgErode = cv.erode(imgDial,kernel,iterations=1)

    return imgErode
    

def getContornosBillete(img):
    imgContornos = img.copy()

    areaMasAdecuada = 0
    contornoMasAdecuado = 0

    altoImg,largoImg,canales = img.shape
    for i in range(3):
        imgEdges = preProcesamiento(img,i)
        contornos, jerarquia = cv.findContours(imgEdges,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) 

        for contorno in contornos:
            perimetro = cv.arcLength(contorno,True)                

            approxEsquinas = cv.approxPolyDP(contorno,0.02*perimetro, True)            

            x, y, w, h = cv.boundingRect(approxEsquinas)
            
            areaEncontrada = w * h
            areaTotal = altoImg * largoImg

            porcentajeW = (w * 100) / largoImg
            porcentajeH = (h * 100) / altoImg
            porcentajeA = (areaEncontrada * 100) / areaTotal
                    
            if altoImg == 512:
                if porcentajeA > 33 and porcentajeA < 83:
                    if porcentajeW > 66 and porcentajeW <= 95:
                        if porcentajeH > 21 and porcentajeH < 89:
                            if areaMasAdecuada < areaEncontrada:
                                contornoMasAdecuado = contorno
                                areaMasAdecuada = areaEncontrada
            elif largoImg == 512:
                if porcentajeA > 37 and porcentajeA < 72:
                    if porcentajeW > 45 and porcentajeW <= 100:
                        if porcentajeH > 64 and porcentajeH < 95:
                            if areaMasAdecuada < areaEncontrada:
                                contornoMasAdecuado = contorno
                                areaMasAdecuada = areaEncontrada
        
    #si no encuentra quitará un porcentaje alrededor de la foto
    if areaMasAdecuada == 0:
        if altoImg == 512:
            quitarW = int(( 7 * largoImg ) / 100)
            quitarH = int(( 20 * altoImg ) / 100)
        elif largoImg == 512:
            quitarW = int(( 20 * largoImg ) / 100)
            quitarH = int(( 7 * altoImg ) / 100)

        retVal = imgContornos[quitarH:altoImg-quitarH,quitarW:largoImg-quitarW]
    else:
        perimetro = cv.arcLength(contornoMasAdecuado,True)
        approx = cv.approxPolyDP(contornoMasAdecuado,0.02*perimetro, True)
        x, y, w, h = cv.boundingRect(approx) 
        retVal = imgContornos[y:(h+y),x:(w+x)]                   

    return retVal

def main(argv):
    foto = cv.imread(argv[0])
    
    img = getContornosBillete(foto)
    cv.imshow("IMG",img)
    cv.waitKey(0)

if __name__ == "__main__":
    main(sys.argv[1:])