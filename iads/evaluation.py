# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy # pour deepcopy()
import psutil
from time import time

def crossval(X, Y, n_iterations, iteration):
    """
    Sépare les données X, Y en un ensemble d'apprentissage et un ensemble de test
    selon la validation croisée.
    Arguments :
    
    - X : numpy array, les données
    - Y : numpy array, les labels
    - n_iterations : int, nombre total de groupes de test
    - iteration : int, numéro de l'itération actuelle (0 <= iteration < n_iterations)
  
    Retourne :
    - Xapp, Yapp : ensembles d'apprentissage
    - Xtest, Ytest : ensemble de test
    """

    n = len(X)
    fold_size = n // n_iterations  # Taille d'un groupe de test
    start = iteration * fold_size
    end = start + fold_size if iteration < n_iterations - 1 else n  # Inclure le reste pour la dernière itération
    test_indices = np.arange(start, end)
    train_indices = np.concatenate((np.arange(0, start), np.arange(end, n)))
    Xtest, Ytest = X[test_indices], Y[test_indices]
    Xapp, Yapp = X[train_indices], Y[train_indices]

    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    """ 
    Réalise une validation croisée stratifiée : 
    - Sépare le dataset en `n_iterations` groupes, tout en respectant la répartition des classes.
    - À l'itération `iteration`, sélectionne 1 groupe pour le test et le reste pour l'apprentissage.
    """
    # Dictionnaire stockant les indices par classe
    class_indices = {}
    
    for i, label in enumerate(Y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    train_indices = []
    test_indices = []

    # Appliquer la validation croisée pour chaque classe
    for label, indices in class_indices.items():
        np.random.shuffle(indices)  # Mélanger les indices de la classe
        
        fold_size = len(indices) // n_iterations  # Taille d'un fold
        start = iteration * fold_size
        end = start + fold_size if iteration < n_iterations - 1 else len(indices)  # Cas spécial pour le dernier fold
        test_indices.extend(indices[start:end])   # Sélection du fold `iteration` pour le test
        train_indices.extend(indices[:start] + indices[end:])  # Le reste va dans l'entraînement

    # Extraction des données
    print(len(train_indices))
    
    Xapp, Yapp = X[train_indices], Y[train_indices]
    Xtest, Ytest = X[test_indices], Y[test_indices]

    return Xapp, Yapp, Xtest, Ytest


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L) , np.std(L) 


def matrice_confusion(Y_true, Y_pred, labels=None):
    """
    Calcule la matrice de confusion 

    Paramètres : 
    ------------
    Y_true : list[int]
        Classes réelles
    Y_pred : list[int]
        Classes predites
    labels : list[int] ou None
        Liste des labels possibles (ex: [0, 1, 2, ...])
        Si None, déduite automatiquement 

    Retour : 
    --------
    np.ndarray : matrice (n_classes * n_classes)
    """
    if labels is None:
        labels = sorted(set(Y_true) | set(Y_pred))
    n = len(labels)
    label_index = {label: i for i, label in enumerate(labels)}

    mat = np.zeros((n, n), dtype=int)

    for yt, yp in zip(Y_true, Y_pred):
        i = label_index[yt]
        j = label_index[yp]
        mat[i,j] += 1
    
    return mat, labels


def afficher_matrice_confusion(mat, labels):
    """
    Affiche la matrice de confusion avec matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(mat, cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    fig.colorbar(cax)

    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Affichage des valeurs
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()























def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    deb = time()
    perf = []
    Y_true_total = []
    Y_pred_total = []
    for i in range(nb_iter):
        Xtrain, Ytrain, Xtest, Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        classifier = copy.deepcopy(C)
        print(f"RAM utilisée : {psutil.virtual_memory().percent}%")
        classifier.train(Xtrain, Ytrain)
        taux_bonne_classif = classifier.accuracy(Xtest,Ytest)
        perf.append(taux_bonne_classif)  
        print(f'Itération {i}: taille base app.={len(Xtrain)} taille base test={len(Xtest)} Taux de bonne classif: {taux_bonne_classif}')
        fin = time()
        print(f"\nTemps total d'un execution : {fin - deb:.2f} s")      
        
        # Prédictions et accumulation
        Y_pred = [classifier.predict(x) for x in Xtest]
        Y_true_total.extend(Ytest)
        Y_pred_total.extend(Y_pred)
    
    fin_final = time()
    print(f"\nTemps total de la validation croisé : {fin_final - deb:.2f} s")      

    taux_moyen, taux_ecart = analyse_perfs(perf)

    return perf, taux_moyen, taux_ecart , Y_true_total, Y_pred_total


