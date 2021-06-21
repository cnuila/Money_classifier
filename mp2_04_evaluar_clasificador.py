import sys
import json
import numpy as np
import joblib
from sklearn import cluster,ensemble

def crearHistograma(kmean, descriptores, k):
    histogramas = np.zeros((len(descriptores),k),"float32")
    for i in range(len(descriptores)):
        predicciones =  kmean.predict(descriptores[i][1])
        for prediccion in predicciones:
            histogramas[i][prediccion] += 1

    return histogramas

def leerNombreFotos(descriptores):
    nombreFotos = []
    for descriptor in descriptores:
        nombreFotos.append(descriptor[0])
    return nombreFotos

def evaluar(descriptores, codeBook,k, clasificador, archivoSalida, nombreFotos):        
    x_prueba = crearHistograma(codeBook,descriptores,k)    
    y_pred = clasificador.predict(x_prueba)

    objeto = {}
    i = 0
    for prediccion in y_pred:
        nombreFoto = nombreFotos[i]
        objeto[nombreFoto] = { "denomiacion":prediccion}
        i+=1

    with open(archivoSalida, "w") as file:
        json.dump(objeto, file, indent=3)


def main(argv):
    descriptores = joblib.load(argv[0])
    nombreFotos = leerNombreFotos(descriptores)
    codeBook, k = joblib.load(argv[1])
    clasificador = joblib.load(argv[2])
    archivoSalida = argv[3]
    
    evaluar(descriptores,codeBook,k, clasificador, archivoSalida, nombreFotos)

if __name__ == "__main__":
    main(sys.argv[1:])