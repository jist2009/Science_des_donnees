import numpy as np
import matplotlib.pyplot as plt

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
