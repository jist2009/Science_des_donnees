# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd

# ------------------------ 

def partitionne(m_desc,m_class, n, s):
    """ input:
        - m_desc : (np.array) matrice des descriptions toutes numériques 
        - m_class : (np.array) matrice des classes 
        - n : (int) numéro de colonne de m_desc
        - s : (float) seuil pour le critère d'arrêt 
    Hypothèse : m_desc peut être partionné ! (il contient au moins 2 valeurs différentes) 
    output : un tuple composé de 2 tuples   
    """
    return ((m_desc[m_desc[:,n] <= s], m_class[m_desc[:,n] <= s]),  (m_desc[m_desc[:,n]> s], m_class[m_desc[:,n]>s]))

def initialise_CHA(df):
    """ DataFrame -> dictionnaire"""
    partition = {}
    for i in range(len(df)):
        partition[i] = [i]
    return partition

def normalisation(df):
    """ DataFrame -> DataFrame normalisée valeur entre 0 et 1"""
    # parcourir pour trouver la valeur max parmi toutes les valeurs
    # simple, il existe une fonction qui récupère valeur min ou max pour une colonne donnée
    df_normalise = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val: 
            df_normalise[col] = (df[col] - min_val) / (max_val - min_val)
        else : 
            df_normalise[col] = 0.0
            print("Impossible de normalisée")
    return df_normalise

def dist_euclidienne(x1, x2) : 
    """ array * array -> float 
    retourne la distance euclidienne entre les deux exemples
    """
    return np.linalg.norm(x1 - x2)
    

def centroide(data):
    """DataFrame ou np.array -> np.array
    Renvoie le centroïde (centre de gravité) de l'ensemble des exemples."""
    if isinstance(data, pd.DataFrame):
        return data.mean()
    elif isinstance(data, np.ndarray):
        return np.mean(data, axis=0)
    else:
        raise TypeError("Le type de donnée doit être pandas.DataFrame ou numpy.array")


##########################################################################################

def dist_centroides(A, B):
    """DataFrame ou np.array -> np.array
    Renvoie distance entre le centroïde de A et B"""
    centroide_A = centroide(A)
    centroide_B = centroide(B)
    return float(dist_euclidienne(centroide_A, centroide_B))

def dist_complete(A, B):
    """ Calcule la distance complete linkage entre deux clusters A et B.
        A et B sont soit des DataFrames, soit des numpy arrays.
    """
    dist_max = 0.0
    for a in range(len(A)):
        for b in range(len(B)):
            # Récupérer les vecteurs a et b
            xa = A.iloc[a].to_numpy() if isinstance(A, pd.DataFrame) else A[a]
            xb = B.iloc[b].to_numpy() if isinstance(B, pd.DataFrame) else B[b]

            d = np.linalg.norm(xa - xb)
            if d > dist_max:
                dist_max = d

    return dist_max

def dist_simple(A, B):
    """ Calcule la distance complete linkage entre deux clusters A et B.
        A et B sont soit des DataFrames, soit des numpy arrays.
    """
    dist_min = 0.0
    for a in range(len(A)):
        for b in range(len(B)):
            # Récupérer les vecteurs a et b
            xa = A.iloc[a].to_numpy() if isinstance(A, pd.DataFrame) else A[a]
            xb = B.iloc[b].to_numpy() if isinstance(B, pd.DataFrame) else B[b]

            d = np.linalg.norm(xa - xb)
            if d < dist_min:
                dist_min = d

    return dist_min

def dist_average(A, B):
    """ Calcule la distance complete linkage entre deux clusters A et B.
        A et B sont soit des DataFrames, soit des numpy arrays.
    """
    dist_average = 0.0
    for a in range(len(A)):
        for b in range(len(B)):
            # Récupérer les vecteurs a et b
            xa = A.iloc[a].to_numpy() if isinstance(A, pd.DataFrame) else A[a]
            xb = B.iloc[b].to_numpy() if isinstance(B, pd.DataFrame) else B[b]

            dist_average += np.linalg.norm(xa - xb)
    
    return dist_average/(len(A)+len(B))

##########################################################################################

def fusionne(df, P0, verbose=False):
    """ Fusionne les deux clusters les plus proches (centroid linkage)

    Args:
        df (DataFrame or np.array): données initiales 
        P0 (dict): partition {clé: [indices]}
        verbose (bool): afficher les opérations 

    Returns:
        (P1, key1, key2, dist): nouvelle partition, les deux clés fusionnées, distance entre eux
    """
    min_dist = float("inf")
    best_pair = (None, None)

    keys = list(P0.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1, key2 = keys[i], keys[j]
            indices1, indices2 = P0[key1], P0[key2]

            # Récupérer les données des clusters
            A = df.iloc[indices1] if isinstance(df, pd.DataFrame) else df[indices1]
            B = df.iloc[indices2] if isinstance(df, pd.DataFrame) else df[indices2]

            # Distance entre centroïdes 
            d = dist_centroides(A, B)

            if d < min_dist:
                min_dist = d
                best_pair = (key1, key2)

    key1, key2 = best_pair
    
    # Création de la nouvelle partition 
    P1 = P0.copy()
    new_key = max(P0.keys()) + 1 
    P1[new_key] = P0[key1] + P0[key2]
    del P1[key1]
    del P1[key2]

    if verbose:
        print(f"fusionne: distance mininimale trouvée entre  [{key1}, {key2}]  =  {min_dist}")
        print(f"fusionne: les 2 clusters dont les clés sont  [{key1}, {key2}]  sont fusionnés")
        print(f"fusionne: on crée la  nouvelle clé {new_key}  dans le dictionnaire.")
        print(f"fusionne: les clés de  [{key1}, {key2}]  sont supprimées car leurs clusters ont été fusionnés.")

    return P1, key1, key2, min_dist


import scipy.cluster.hierarchy
import matplotlib as plt
import scipy.cluster.hierarchy

def CHA_centroid(df, verbose=False, dendrogramme=False):
    """ Applique le clustering hiérarchique ascendant avec centroid linkage.

    Args:
        df (DataFrame ou ndarray): les données
        verbose (bool): si True, affiche les étapes avec messages
        dendrogramme (bool): si True, affiche le dendrogramme

    Returns:
        list: liste des fusions sous forme [clé1, clé2, distance, taille fusionnée]
    """
    # Initialisation de la partition
    partition = initialise_CHA(df)
    fusions = []

    while len(partition) > 1:
        partition, key1, key2, dist = fusionne(df, partition, verbose=verbose)

        new_key = max(partition.keys())
        taille_fusion = len(partition[max(partition.keys())])  # le nouveau cluster vient d’être créé
        fusions.append([key1, key2, dist, taille_fusion])

        if verbose:
            print(f"CHA_centroid: une fusion réalisée de  {key1}  avec  {key2} de distance  {dist:.4f}")
            print(f"CHA_centroid: le nouveau cluster contient  {taille_fusion}  exemples")

    if verbose:
        print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme:
        # On convertit la liste de fusions au format numpy attendu par scipy
        Z = np.array(fusions)

        # Affichage du dendrogramme
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(Z, leaf_font_size=24.)   # taille des caractères de l'axe des X
        plt.show()
        
    return fusions



def fusionne_custom(df, P0, dist_func, verbose=False):
    min_dist = float('inf')
    best_pair = (None, None)

    keys = list(P0.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key1, key2 = keys[i], keys[j]
            A = df.iloc[P0[key1]] if isinstance(df, pd.DataFrame) else df[P0[key1]]
            B = df.iloc[P0[key2]] if isinstance(df, pd.DataFrame) else df[P0[key2]]

            d = dist_func(A, B)
            if d < min_dist:
                min_dist = d
                best_pair = (key1, key2)

    key1, key2 = best_pair
    new_key = max(P0.keys()) + 1

    if verbose:
        print(f"fusionne: distance mininimale trouvée entre  [{key1}, {key2}]  =  {min_dist}")
        print(f"fusionne: les 2 clusters dont les clés sont  [{key1}, {key2}]  sont fusionnés")
        print(f"fusionne: on crée la  nouvelle clé {new_key}  dans le dictionnaire.")
        print(f"fusionne: les clés de  [{key1}, {key2}]  sont supprimées car leurs clusters ont été fusionnés.")

    P1 = P0.copy()
    P1[new_key] = P0[key1] + P0[key2]
    del P1[key1]
    del P1[key2]

    return P1, key1, key2, min_dist
    
# je ne comprends pas la question, il demnade de faire, autant créer une fonction CHA abstraite 
# puis faire une déclinaisons sur CHA_complete, CHA_simple, CHA_average





def CHA_custom(df, dist_func, verbose=False, dendrogramme=False):
    if verbose:
        print("CHA_custom: clustering hiérarchique avec fonction de distance personnalisée")
    partition = initialise_CHA(df)
    fusions = []

    while len(partition) > 1:
        partition, key1, key2, dist = fusionne_custom(df, partition, dist_func, verbose=verbose)
        new_key = max(partition.keys())
        taille_fusion = len(partition[new_key])
        fusions.append([key1, key2, dist, taille_fusion])
        if verbose:
            print(f"CHA_custom: fusion {key1} & {key2} -> distance = {dist:.4f}, taille = {taille_fusion}")


    if dendrogramme:
        # On convertit la liste de fusions au format numpy attendu par scipy
        Z = np.array(fusions)

        # Affichage du dendrogramme
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(Z, leaf_font_size=24.)   # taille des caractères de l'axe des X
        plt.show()
            
    return fusions

def afficher_dendrogrammes(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    methodes = [
        ("Centroid", CHA_centroid),
        ("Complete", CHA_complete),
        ("Simple", CHA_simple),
        ("Average", CHA_average)
    ]

    for ax, (nom, methode) in zip(axs.ravel(), methodes):
        fusions = methode(df)
        linkage_mat = to_linkage_format(fusions)
        dendrogram(linkage_mat, ax=ax)
        ax.set_title(f"Dendrogramme - {nom} linkage")

    plt.tight_layout()
    plt.show()
# Déclinaisons
def CHA_complete(df, verbose=False):
    return CHA_custom(df, dist_complete, verbose)

def CHA_simple(df, verbose=False):
    return CHA_custom(df, dist_simple, verbose)

def CHA_average(df, verbose=False):
    return CHA_custom(df, dist_average, verbose)



def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  Clustering hiérarchique ascendant générique.

    Args:
        DF (DataFrame ou ndarray): données
        verbose (bool): affichage des étapes
        linkage (str): type de linkage parmi "centroid", "complete", "average"

    Returns:
        list: liste des fusions sous forme [clé1, clé2, distance, taille fusionnée]
    """
    if verbose:
        print(f"CHA: clustering hiérarchique avec linkage = '{linkage}'")
    
    # Dictionnaire des fonctions de distance
    fonction_linkage = {
        "centroid": dist_centroides,
        "complete": dist_complete,
        "simple": dist_simple,
        "average": dist_average
    }
    
    fonction_distance = fonction_linkage[linkage] 

    return CHA_custom(DF, fonction_distance, verbose)