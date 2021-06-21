import sys
import os
import cv2 as cv
import numpy as np
import csv
import time
from sklearn import cluster
import joblib

def cargarImagenes(directorio):
    fotos = []
    cantArchivos = 0
    for archivo in sorted(os.listdir(directorio)):
        if cantArchivos == 5:
            break
        img = cv.imread(os.path.join(directorio,archivo))
        fotos.append((archivo, getContornosBillete(img)))
        cantArchivos+=1
    return (fotos, cantArchivos)

def autoCanny(img):
    medianaVector = np.median(img)
    sigma = 0.33
    lowerThre = int(max(0,(1.0 - sigma) * medianaVector))
    upperThre = int(min(255,(1.0 + sigma) * medianaVector))
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

    if largoImg == 512:
        retVal= cv.rotate(retVal, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return retVal

#funcion que retorna los descriptores de los puntos clave de la foto
def getDescriptores(fotos, esPrueba):
    brisk = cv.BRISK_create(30)
    descriptores = []
    #guardar descriptores
    for nombreArchivo,foto in fotos:
        puntosClave, descriptoresActuales = brisk.detectAndCompute(foto,None)            
        descriptores.append((nombreArchivo,descriptoresActuales))
    #convertir a 1 fila
    descriptoresFila = descriptores[0][1]
    for nombreArchivo, descriptor in descriptores[1:]:
        descriptoresFila = np.vstack((descriptoresFila, descriptor))
    
    #convertir a float
    descriptoresFloat = descriptoresFila.astype(float)
    
    return (descriptores, descriptoresFloat)

#funcion que guarda en fit los visual words y luego predice cada uno de los descriptores
def kMeans(descriptores, descriptoresFloat, esPrueba, archivoCodeBook, cantidadArchivos):        
    if esPrueba == 0:
        k = 64
        kmean = cluster.KMeans(n_clusters=k)
        kmean = kmean.fit(descriptoresFloat)
        joblib.dump((kmean, k),archivoCodeBook,compress=3)
    else:
        kmean, k = joblib.load(archivoCodeBook)
    
    histogramas = np.zeros((cantidadArchivos,k),"float32")
    for i in range(cantidadArchivos):
        predicciones =  kmean.predict(descriptores[i][1])
        for prediccion in predicciones:
            histogramas[i][prediccion] += 1

    return histogramas

def extraerCaracteristicas(fotos, archivoSalida, esPrueba, archivoCodeBook, cantArchivos):    
    descriptores, descriptoresFloat = getDescriptores(fotos,esPrueba)
    histogramas = kMeans(descriptores, descriptoresFloat, esPrueba,archivoCodeBook,cantArchivos)

    with open(archivoSalida, "w") as file:
        writer = csv.writer(file)
        writer.writerows(histogramas)

def main(argv):
    fotos, cantFotos = cargarImagenes(argv[0])
    esPrueba  = int(argv[2])
    timeInicio = time.time()      
    extraerCaracteristicas(fotos, argv[1], esPrueba, argv[3],cantFotos)        
    timeFinal = time.time()
    timeT = timeFinal - timeInicio
    print("Le tomo %s segundos" % (timeT)) 

if __name__ == "__main__":
    main(sys.argv[1:])