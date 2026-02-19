# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2025

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iads.Classifiers import Classifier


# ------------------------ 

def genere_dataset_uniform(min,max,nb):
    vect_desc = np.random.uniform(binf,bsup,(2*n,d))
    vect_label = np.array([-1]*n + [1]*n)
    return (vect_desc,vect_label)

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    list_positive = np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    list_negative = np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    list_normal = np.concatenate((list_negative,list_positive))
    list_label = np.array([-1]*nb_points + [1]*nb_points)
    return (list_normal,list_label)

def plot2DSet(desc,labels,nom_dataset= "Dataset", avec_grid=False):    
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]

    # Tracé de l'ensemble des exemples : ok
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red", label='classe -1') # 'o' rouge pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue", label='classe +1') # 'x' bleu pour la classe +1

    # Informations d'affichage :
    plt.title(nom_dataset)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()  # Grille: à mettre, ou pas
    
    # Visualisation du résultat
    plt.show()
    return 

def plot_frontiere(desc_set, label_set, classifier, step=30):
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
    return 


def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    centres = np.array([[0,0], [1, 1], [1, 0], [0, 1]])
    labels = np.array([-1, -1, 1, 1])

 #  Matrice de covariance isotrope (var sur les deux axes)
    sigma = np.array([[var, 0], [0, var]])

    # Génération des points selon une distribution gaussienne
    x = np.vstack([np.random.multivariate_normal(mean=centres[i], cov=sigma, size=n) for i in range(4)])    
    y = np.hstack([np.full(n, labels[i]) for i in range(4)])

    return x, y















