import hdbscan
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import csv
import seaborn as sns
import os
plt.ioff()

def readIrisData():
    '''
    Data Format:
    1. sepal length in cm 
    2. sepal width in cm 
    3. petal length in cm 
    4. petal width in cm 
    5. class: 
        -- Iris Setosa 
        -- Iris Versicolour 
        -- Iris Virginica
    '''
    data = []
    with open('/mnt/c/Users/mokho_000/Bash/python/iris.data.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
#                print(row)
                data.append(row)
    #for some reason it adds empty item to end of list
    del data[-1]
    
    print("Read data from Iris dataset complete.")

    return data

def renderClusters(clusterer, data1, data2, title=None):
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    color_palette = sns.color_palette("bright", max(labels) + 2)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.75, 0.75, 0.75)
                      for x in labels]
    cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, probabilities)])   
    fig = plt.figure(1)
    plt.scatter(data1[0:50], data2[0:50], s=15, marker='s', 
                c=cluster_member_colors[0:50])
    plt.scatter(data1[51:100], data2[51:100], s=15, marker='*', 
                c=cluster_member_colors[51:100])
    plt.scatter(data1[100:], data2[100:], s=15, marker='x', 
                c=cluster_member_colors[100:])

    plt.scatter([], [], label="Iris Setosa", marker='s', c="grey")
    plt.scatter([], [], label="Iris Versicolour", marker='*', c="grey")
    plt.scatter([], [], label="Iris Virginica", marker='x', c="grey")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                ncol=3, borderaxespad=0.)
    if title is not None:
        out_png = (os.getcwd() + "/" + title + ".png")
    else:
        out_png = '/mnt/c/Users/mokho_000/Bash/python/test.png'
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    print("Clusters rendered and saved to " + os.getcwd() + " as .png")

iris = readIrisData()

data_sepal_len = [x[0] for x in iris]
data_sepal_wid = [x[1] for x in iris]
data_petal_len = [x[2] for x in iris]
data_petal_wid = [x[3] for x in iris]
data_classname = [x[4] for x in iris]

sepal_data = zip(data_sepal_len, data_sepal_wid)

clusterer = hdbscan.HDBSCAN(cluster_selection_method="leaf", 
                            min_cluster_size=25, 
                            min_samples=1).fit(sepal_data)

renderClusters(clusterer, data_sepal_len, data_sepal_wid, title="SepalClust")

petal_data = zip(data_petal_len, data_petal_wid)

petalClusterer = hdbscan.HDBSCAN(cluster_selection_method="leaf", 
                            min_cluster_size=25, 
                            min_samples=1).fit(petal_data)

renderClusters(petalClusterer, data_petal_len, data_petal_wid, title="PetalClust")

all_data = zip(data_sepal_len, data_sepal_wid,data_petal_len, data_petal_wid)

clusterer4D = hdbscan.HDBSCAN(cluster_selection_method="leaf", 
                            min_cluster_size=25, 
                            min_samples=1).fit(all_data)

renderClusters(clusterer4D, data_sepal_len, data_sepal_wid, title="4D-Clustering")
