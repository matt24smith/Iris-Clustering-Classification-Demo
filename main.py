"""
A short demonstration of AI pattern recognition and hierarchical clustering,
in 100 lines of code.

Author: Matt Smith
"""

import hdbscan
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    with open('/mnt/c/Users/mokho_000/Bash/python/iriscluster/iris.data.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
#                print(row)
                data.append(row)
    # to deal with extra datum created by newline char
    del data[-1]
    
    print("Read data from Iris dataset complete.")

    return data

def renderClusters(clusterer, x, y, z, title=None):
    """
    Generate figures and save as .png
    """
    data1=np.asarray(x, dtype='float64')
    data2=np.asarray(y, dtype='float64')
    data3=np.asarray(z, dtype='float64')
    nulldata=np.asarray([], dtype='float64')

    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    color_palette = sns.color_palette("bright", max(labels) + 2)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.75, 0.75, 0.75)
                      for x in labels]
    cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, probabilities)])   
    fig = plt.figure(1)
    ax=fig.add_subplot(111, projection='3d')
    ax.autoscale_view(tight=True)

    ax.scatter(data1[0:50], data2[0:50], data3[0:50], s=15, marker='s', 
                c=cluster_member_colors[0:50])
    ax.scatter(data1[51:100], data2[51:100], data3[51:100], s=15, marker='*', 
                c=cluster_member_colors[51:100])
    ax.scatter(data1[100:], data2[100:], data3[100:], s=15, marker='x', 
                c=cluster_member_colors[100:])

    ax.scatter(nulldata, nulldata, nulldata, label="Iris Setosa", marker='s', c="grey")
    ax.scatter(nulldata, nulldata, nulldata, label="Iris Versicolour", marker='*', c="grey")
    ax.scatter(nulldata, nulldata, nulldata, label="Iris Virginica", marker='x', c="grey")
    ax.scatter(nulldata, nulldata, nulldata, 
               label="\nNot pictured: petal\nlength on 4th axis", marker='')

    ax.legend(loc = 'upper right', bbox_to_anchor=(1,1.), ncol=1, 
                       fancybox=True, shadow=True)
    
    plt.title('Iris Clustering Classification')
    ax.set_xlabel('Iris Sepal Length')
    ax.set_ylabel('Iris Sepal Width')
    ax.set_zlabel('Iris Petal Width')
    
    if title is None:
        title = "test"
    out_png = (os.getcwd() + "/" + title + ".png")
    plt.savefig(out_png, dpi=250)
    plt.close()

    print("Clusters rendered and saved to " + os.getcwd() + " as .png")

iris = readIrisData()

data_sepal_len = [x[0] for x in iris]
data_sepal_wid = [x[1] for x in iris]
data_petal_len = [x[2] for x in iris]
data_petal_wid = [x[3] for x in iris]
data_classname = [x[4] for x in iris]

iris_data = zip(data_sepal_len, data_sepal_wid,data_petal_len, data_petal_wid)

clusterer4D = hdbscan.HDBSCAN(cluster_selection_method="leaf", 
                            min_cluster_size=25, 
                            min_samples=1).fit(iris_data)

renderClusters(clusterer4D, data_sepal_len, data_sepal_wid, data_petal_len, 
               title="4D-Clustering")
