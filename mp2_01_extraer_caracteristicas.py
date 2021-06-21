import sys
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plot
import joblib

def cargarImagenes(directorio):
    fotos = []
    i = 0
    for archivo in sorted(os.listdir(directorio)):
        img = cv.imread(os.path.join(directorio,archivo))
        fotos.append((archivo, getContornosBillete(img)))
    return fotos

def autoCanny(img):
    medianaVector = np.median(img)
    sigma = 0.33
    lowerThre = int(max(0, (1.0 - sigma) * medianaVector))
    upperThre = int(min(255, (1.0 + sigma) * medianaVector))
    imgCanny = cv.Canny(img, lowerThre, upperThre)
    return imgCanny

# 0 wide, 1 tight, 2 automatico con mediana
def preProcesamiento(img, numCanny):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (3, 3), 1)

    if numCanny == 0:
        imgCanny = cv.Canny(imgBlur, 10, 200)
    elif numCanny == 1:
        imgCanny = cv.Canny(imgBlur, 255, 250)
    else:
        imgCanny = autoCanny(imgBlur)

    kernel = np.ones((2, 2))
    imgDial = cv.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv.erode(imgDial, kernel, iterations=1)

    return imgErode

def getContornosBillete(img):
    imgContornos = img.copy()

    areaMasAdecuada = 0
    contornoMasAdecuado = 0

    altoImg, largoImg, canales = img.shape
    for i in range(3):
        imgEdges = preProcesamiento(img, i)
        contornos, jerarquia = cv.findContours(
            imgEdges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for contorno in contornos:
            perimetro = cv.arcLength(contorno, True)

            approxEsquinas = cv.approxPolyDP(contorno, 0.02*perimetro, True)

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
            quitarW = int((7 * largoImg) / 100)
            quitarH = int((20 * altoImg) / 100)
        elif largoImg == 512:
            quitarW = int((20 * largoImg) / 100)
            quitarH = int((7 * altoImg) / 100)

        retVal = imgContornos[quitarH:altoImg -
                              quitarH, quitarW:largoImg-quitarW]

    else:
        perimetro = cv.arcLength(contornoMasAdecuado, True)
        approx = cv.approxPolyDP(contornoMasAdecuado, 0.02*perimetro, True)
        x, y, w, h = cv.boundingRect(approx)
        retVal = imgContornos[y:(h+y), x:(w+x)]

    if largoImg == 512:
        retVal = cv.rotate(retVal, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

    if largoImg == 512:
        retVal= cv.rotate(retVal, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return retVal

def loadModels():
    billete1 = cv.imread("Modelos/1 Lempira.jpg")
    billete2 = cv.imread("Modelos/2 Lempiras.jpg")
    billete5 = cv.imread("Modelos/5 Lempiras.jpg")
    billete10 = cv.imread("Modelos/10 Lempiras.jpg")
    billete20 = cv.imread("Modelos/20 Lempiras.jpg")
    billete50 = cv.imread("Modelos/50 Lempiras.jpg")
    billete100 = cv.imread("Modelos/100 Lempiras.jpg")
    billete500 = cv.imread("Modelos/500 Lempiras.jpg")

    billete1HSV = cv.cvtColor(billete1, cv.COLOR_BGR2HSV)
    billete2HSV = cv.cvtColor(billete2, cv.COLOR_BGR2HSV)
    billete5HSV = cv.cvtColor(billete5, cv.COLOR_BGR2HSV)
    billete10HSV = cv.cvtColor(billete10, cv.COLOR_BGR2HSV)
    billete20HSV = cv.cvtColor(billete20, cv.COLOR_BGR2HSV)
    billete50HSV = cv.cvtColor(billete50, cv.COLOR_BGR2HSV)
    billete100HSV = cv.cvtColor(billete100, cv.COLOR_BGR2HSV)
    billete500HSV = cv.cvtColor(billete500, cv.COLOR_BGR2HSV)

    l1= np.array([0, 0, 0])
    u1= np.array([9, 255, 255])
    mask1 = cv.inRange(billete1HSV,l1,u1)
    result1 = cv.bitwise_and(billete1, billete1, mask=mask1)

    l2= np.array([113, 8, 0])
    u2= np.array([173, 255, 255])
    mask2 = cv.inRange(billete2HSV,l2,u2)
    result2 = cv.bitwise_and(billete2, billete2, mask=mask2)

    l5= np.array([13, 0, 0])
    u5= np.array([179, 35, 174])
    mask5 = cv.inRange(billete5HSV,l5,u5)
    result5 = cv.bitwise_and(billete5, billete5, mask=mask5)

    l10= np.array( [0, 20, 0])
    u10= np.array( [179, 255, 186])
    mask10 = cv.inRange(billete10HSV,l10,u10)
    result10 = cv.bitwise_and(billete10, billete10, mask=mask10)

    l20= np.array( [23, 0, 0])
    u20= np.array([179, 255, 255])
    mask20 = cv.inRange(billete20HSV,l20,u20)
    result20 = cv.bitwise_and(billete20, billete20, mask=mask20)

    l50= np.array([24, 0, 0])
    u50= np.array([179, 255, 255])
    mask50 = cv.inRange(billete50HSV,l50,u50)
    result50 = cv.bitwise_and(billete50, billete50, mask=mask50)

    l100= np.array([0, 28, 0])
    u100= np.array([28, 255, 255])
    mask100 = cv.inRange(billete100HSV,l100,u100)
    result100 = cv.bitwise_and(billete100, billete100, mask=mask100)
    
    l500= np.array([89, 24, 31])
    u500= np.array([179, 255, 215])
    mask500 = cv.inRange(billete500HSV,l500,u500)
    result500 = cv.bitwise_and(billete500, billete500, mask=mask500)

    arrayPics = [result1, result2, result5, result10,
                 result20, result50, result100, result500]
    arrayPics2 = [billete1,billete2,billete5,billete10,billete20,billete50,billete100,billete500]
    arrayModels = []

    for model in arrayPics2:
        histModel = cv.calcHist(model, [0], None, [255], [1, 255])
        histModel = cv.normalize(histModel, histModel).flatten()
        arrayModels.append(histModel)
    return arrayModels

def findColor(img):
    arrayModels = loadModels()
    arrayNames = ["1l", "2l", "5l", "10l", "20l", "50l", "100l", "500l"]

    compArray = np.zeros(8)

    billeteHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    l = np.array([6, 12, 1])
    u = np.array([179, 187, 210])
    mask = cv.inRange(billeteHSV,l,u)
    result = cv.bitwise_and(img,img, mask=mask)

    cv.imshow("foto a evaluar", result)
    cv.waitKey(2000)

    histBillete = cv.calcHist(result, [0], None,  [255], [1, 255])
    histBillete = cv.normalize(histBillete, histBillete).flatten()

    cont = 0
    for histModel in arrayModels:
        metricV = cv.compareHist(histBillete, histModel, cv.HISTCMP_INTERSECT)
        compArray[cont] = metricV
        cont += 1

        plot.plot(histModel)
        plot.plot(histBillete)
        plot.show()

    distribution = compArray/np.sum(compArray)
    print("Comparison Array:")
    print(compArray)
    print("Distribution Array: ")
    print(distribution)

    total_objects = 8
    label_objects = ('1', '2', '5', '10', '20', '50', '100', '500')
    font_size = 20
    width = 0.5
    plot.barh(np.arange(total_objects), distribution, width, color='r')
    plot.yticks(np.arange(total_objects) + width/2.,
                label_objects, rotation=0, size=font_size)
    plot.xlim(0.0, 1.0)
    plot.ylim(-0.5, 8.0)
    plot.xlabel('Probability', size=font_size)

    plot.show()

def empty(a):
    pass

#funcion que retorna los descriptores de los puntos clave de la foto
def getDescriptores(fotos, archivoDescriptores):
    brisk = cv.BRISK_create(30)
    descriptores = []
    #guardar descriptores
    for nombreArchivo,foto in fotos:
        puntosClave, descriptoresActuales = brisk.detectAndCompute(foto,None)            
        descriptores.append((nombreArchivo,descriptoresActuales))

    joblib.dump(descriptores,archivoDescriptores,compress=3)

''''def extraerCaracteristicas(fotos, archivoSalida, esPrueba, archivoCodeBook, cantArchivos):    
    descriptores, descriptoresFloat = getDescriptores(fotos,esPrueba)
    histogramas = kMeans(descriptores, descriptoresFloat, esPrueba,archivoCodeBook,cantArchivos)

    with open(archivoSalida, "w") as file:
        writer = csv.writer(file)
        writer.writerows(histogramas)'''

def main(argv):
    fotos = cargarImagenes(argv[0])
    getDescriptores(fotos,argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])
