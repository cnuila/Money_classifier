import sys
import pandas as pd
import json
import numpy as np
import joblib
from sklearn import ensemble, utils, metrics, cluster
from random import randint

#calcula las estadisticas de cada clase (precision, recall, f1-score)
def estadisticasPorClase(y_prueba, y_pred):

    matrizConfusion = metrics.confusion_matrix(y_prueba,y_pred)
    
    nuevaMatriz = [[0,0],[0,0]]
    clase = "1"
    for i in range(8):
        pos11 = matrizConfusion[i][i]
        pos00 = matrizConfusion[0][0] + matrizConfusion[1][1] + matrizConfusion[2][2] + matrizConfusion[3][3] - pos11                
        pos01 = 0
        pos10 = 0
        for j in range(8):
            pos10 += matrizConfusion[i][j]
            pos01 += matrizConfusion[j][i]
        pos01 -= pos11
        pos10 -= pos11
        nuevaMatriz[0][0] = pos00
        nuevaMatriz[0][1] = pos01
        nuevaMatriz[1][0] = pos10
        nuevaMatriz[1][1] = pos11

        recall = 0
        precision = 0
        f1 = 0
        if (pos11 + pos10) != 0:
            recall = (pos11 / (pos11 + pos10)) * 100    
        if (pos11 + pos01) != 0:
            precision = (pos11 / (pos11 + pos01)) * 100
        if (precision + recall) != 0:
            f1 = round((2 * precision * recall) / (precision + recall), 5)
        
        if i == 1:
            clase = "2"
        elif i == 2:
            clase = "5"
        elif i == 3:
            clase = "10"
        elif i == 4:
            clase = "20"
        elif i == 5:
            clase = "50"
        elif i == 6:
            clase = "100"
        elif i == 7:
            clase = "500"


        print()
        print("--------------------------------------------")
        print(clase)
        imprimirMatrizConfusion(clase,nuevaMatriz)
        print("Recall:  %s" % (round(recall,5)))
        print("Precision:  %s" % (round(precision,5)))
        print("F1-Score:  %s" % (f1))
    
    print()
    print("--------------------------------------------")
    #recall promedio
    print("Recall Promedio: ",round(metrics.recall_score(y_prueba,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")*100,5))
    #preciosion promedio
    print("Precision Promedio: ",round(metrics.precision_score(y_prueba,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")*100,5))
    #f1-socre promedio
    print("F1-Score Promedio: ",round(metrics.f1_score(y_prueba,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")*100,5))

def imprimirMatrizConfusion(clase, matriz):
    print("             Sobrantes   %s" % (clase))
    cont = 0
    for fila in matriz:
        if cont == 0:
            print("Sobrantes       ",end="")
        else:
            print("%s          " % (clase),end="")
        for columna in fila:
            print("%s         " % (columna),end="")
        print("")
        cont+=1 

def crossValidation(modelo, x , y, inFolds, k):
    x_aleatoria, y_aleatoria = utils.shuffle(x, y)
    f1_score_global = 0
    accuracy_global = 0
    cantElem = int(len(x) / inFolds)

    for i in range(0,inFolds):
        print()
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Iteracion :",(i+1))

        validacion_x = x_aleatoria[(i*cantElem):(i*cantElem + cantElem)]            
        validacion_y = y_aleatoria[(i*cantElem):(i*cantElem + cantElem)]

        training_x = x_aleatoria[:(i*cantElem)] + x_aleatoria[(i*cantElem + cantElem):]
        training_y = y_aleatoria[:(i*cantElem)] + y_aleatoria[(i*cantElem + cantElem):]        

        codeBook = kMeans(training_x,k)

        validacion_x = crearHistograma(codeBook,validacion_x,k)
        training_x = crearHistograma(codeBook,training_x,k)

        modelo.fit(training_x,training_y)
        y_pred = modelo.predict(validacion_x)

        f1_score = metrics.f1_score(validacion_y,y_pred,labels=["1","2","5","10","20","50","100","500"],average="macro")        
        accuracy = metrics.accuracy_score(validacion_y,y_pred)        

        estadisticasPorClase(validacion_y,y_pred)
        print("Score:",f1_score)
        print("Accuracy:",accuracy)
        if not pd.isna(f1_score):
            f1_score_global += f1_score
        if not pd.isna(accuracy):
            accuracy_global += accuracy

    f1_score_global /= inFolds
    print("Accuracy Global = ",accuracy_global / inFolds)
    return f1_score
           
#funcion que genera una iteracion de los paramtros unica
def generarIteracionUnica(iteraciones):
    listaMaxDepth = [3, 4, 5, 6, 7, 8, 9, 10, 11, None]
    criterios = ["gini", "entropy"]

    maxDepth = listaMaxDepth[randint(0, 9)]
    nEstimators = randint(40, 80)
    criterion = criterios[randint(0, 1)]
    maxFeatures = randint(1, 6)
    k = randint(1,32) * 64

    nuevaIteracion = [nEstimators, maxDepth, criterion, maxFeatures, k]

    while not esIteracionUnica(iteraciones, nuevaIteracion):
        maxDepth = listaMaxDepth[randint(0, 9)]
        nEstimators = randint(40, 80)
        criterion = criterios[randint(0, 1)]
        maxFeatures = randint(1, 6)
        k = randint(1,32) * 64
        nuevaIteracion = [nEstimators, maxDepth, criterion, maxFeatures, k]

    return nuevaIteracion

#funcion que revisa que los parametros usados sean diferentes
def esIteracionUnica(iteraciones, nuevaIteracion):
    for fila in iteraciones:
        esIgual = True
        for index, columna in enumerate(fila):
            if nuevaIteracion[index] != columna:
                esIgual = False
        if esIgual:
            return False
    return True

#ajustar los hiper parametros con n-fold cross-validation
def busquedaParametros(descriptores,y_entrenamiento,inFolds):
    iteraciones = []   

    mayorPromedio = 0.605
    nuevoPromedio = 0
    cont = 1

    while nuevoPromedio < mayorPromedio:      
        iteracion = generarIteracionUnica(iteraciones)   
        iteraciones.append(iteracion)

        randomForest = ensemble.RandomForestClassifier(n_estimators=iteracion[0],max_depth=iteracion[1],criterion=iteracion[2],max_features=iteracion[3])        
        print("---------------------------------------------")
        print("Configuracion ",cont)
        cont+=1
        score = crossValidation(randomForest,descriptores, y_entrenamiento,inFolds, iteracion[4])
        print(iteracion)
        print("Promedio:",score)
        nuevoPromedio = score    
        if nuevoPromedio > mayorPromedio:
            mejorIteracion = iteracion
                
    print()
    print("------------------------------------")
    print("Mayor Promedio:",nuevoPromedio)
    print("Mejor Iteracion:",mejorIteracion)

#leer las clases de cada foto
def leerJSON(nombreArchivo):
    clases = []
    with open(nombreArchivo, 'r') as file:
        jsonData = file.read()
        etiquetas = json.loads(jsonData)
        for etiqueta in etiquetas:
            clases.append(etiquetas[etiqueta]["denominacion"])
    return clases

#funcion que guarda en fit los visual words y luego predice cada uno de los descriptores
def kMeans(descriptores, k):        

    #convertir a 1 fila
    descriptoresFila = descriptores[0][1]
    for nombreArchivo, descriptor in descriptores[1:]:
        descriptoresFila = np.vstack((descriptoresFila, descriptor))

    #convertir a float
    descriptoresFloat = descriptoresFila.astype(float)

    kmean = cluster.MiniBatchKMeans(n_clusters=k)
    kmean = kmean.fit(descriptoresFloat)

    return kmean
        
def crearHistograma(kmean, descriptores, k):
    histogramas = np.zeros((len(descriptores),k),"float32")
    for i in range(len(descriptores)):
        predicciones =  kmean.predict(descriptores[i][1])
        for prediccion in predicciones:
            histogramas[i][prediccion] += 1

    return histogramas

def main(argv):          

    descriptores = joblib.load(argv[0])
    inEtiquetas = leerJSON(argv[1])
    inFolds = int(argv[2])  

    busquedaParametros(descriptores,inEtiquetas,inFolds)    


if __name__ == "__main__":
    main(sys.argv[1:])