import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster

path = './Imagenes/*.png'
paths = glob.glob(path)
imagenes = []
for i in paths: imagenes.append(plt.imread(i))
imagenes = np.array(imagenes)

X = imagenes.reshape((87,-1))

inercias = []
for i in range(20):
    k_means = sklearn.cluster.KMeans(n_clusters=i+1)
    k_means.fit(X)
    inercias.append(k_means.inertia_)
    
plt.figure(figsize = (10,5))
plt.plot(np.array(range(20))+1,inercias)
plt.xlabel('Número de clusters')
plt.ylabel('Inercias')
plt.title('El mejor número de clusters es 4')
plt.savefig('inercia.png')

k_means = sklearn.cluster.KMeans(n_clusters=4)
k_means.fit(X)
centros = k_means.cluster_centers_

cluster = k_means.predict(X)

indices = []
for i in range(4):
    select = cluster==(i)
    X_select = X[select]
    resta = X_select - centros[i]
    distancias = np.sum(resta**2, axis = 1)
    distancias_original = list(distancias.copy())
    distancias.sort()
    distancias2 = distancias[:5]
    indices_Agregar = []
    for j in distancias2:
        indices_Agregar.append(X_select[distancias_original.index(j)])
    indices.append(indices_Agregar)
    
plt.figure(figsize = (20,15))
subplot = 1
for i in range(4):
    for j in range(5):
        plt.subplot(4,5,subplot)
        plt.imshow(indices[i][j].reshape(100,100,3))
        subplot = subplot + 1
plt.savefig('ejemplo_clases.png')