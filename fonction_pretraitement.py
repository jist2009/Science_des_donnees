import pandas as pd

import string 



def organiser(df):  
    """
    Organise les documents textuels par groupe (ou catégorie) en fonction de leurs labels (targets)
    
    Paramètres: 
    df : pandas.DataFrame 
        DataFrame contenant les colonnes "target" (entier de 0 à 19)
        et "mots_filtre" (liste de mots nettoyés pour chaque document)
    Retour : 
        La fonction retourne 'groupes' = dictionnaire de texte = dict * vect[mot]
        groupes : dict[int, list[list[str]])
    """
    groupes = {i: [] for i in range(20)}  # initialise les 20 groupes
    
    for _, row in df.iterrows():
        groupe = row['target']
        message = pd.Series(row['mots_filtre'])   
        groupes[groupe].append(message)
        
    return groupes 
# Maintenant 'groupes' contient : pour chaque numéro, la liste des messages


def filtre(df, stop_words):
    """ 
    Nettoie chaque ligne textuelle d'un DataFrame et ajoute une nouvelle colonne 'mots_filtre' 
    contenant la liste des mots filtrés

    Paramètre :
    -----------
    df : pandas.DataFrame
        un DataFrame dont la première colonne contient des chaînes de caractères (textes à nettoyer)
    stop_words : pandas.DataFrame
        un DataFrame contenant la liste de mots vides stockée dans une seule colonne

    Retour : 
    --------
    pandas.DataFrame 
        Le DataFrame d'origine, avec une colonne supplémentaire "mots_filtre"
    
    """
    stop_words = set(stop_words.iloc[:,0])
    
    def nettoyer_ligne(ligne):
        if pd.isna(ligne):
            return []  # Ligne vide → on retourne une liste vide
        
        ligne = ligne.replace("'", " ")
        ligne = ligne.translate(str.maketrans('', '', string.punctuation))
        ligne = ligne.lower()
        mots = ligne.split()
        mots_filtres = []
        for mot in mots:
            if mot in stop_words:
                continue 
            mots_filtres.append(mot)
        return mots_filtres

    # On applique la fonction à chaque ligne et on stocke le résultat dan s une nouvelle colonne
    df['mots_filtre'] = df.iloc[:, 0].apply(nettoyer_ligne)
    
    return df


import math
import numpy as np
from collections import defaultdict, Counter

def tf_idf(X):
    """
    Calcule les poids TF-IDF des mots dans chaque groupe de message.

    Paramètre :
    -----------
    X : list[list[str]]
        Liste de message, chaque message étant une liste de mots.
   
    Retour :
    --------
    tfidf : dict[str, float]]
        Un dictionnaire donnant le score TF-IDF de chaque mot.
    """
    tfidf = defaultdict(float)
    N = len(X)  # nombre total de documents
    DF = defaultdict(int)  # document frequency : mot -> nombre de documents contenant ce mot


    # 1. Compter DF (Document Frequency) sur tout le corpus
    for doc in X:
        for word in set(doc):
            DF[word] += 1

    # 2. Calculer TF-IDF pour chaque groupe
    for message in X:
        tf = Counter(message)
        doc_len = len(message)
        for word, freq in tf.items():
            tf_score = freq / doc_len
            idf_score = math.log(N / (DF[word] + 1))
            tfidf[word] += tf_score * idf_score  # somme sur tous les docs du groupe
    """
    # 3. Moyenne sur tous les documents 
    for word in tfidf:
        tfidf[word] /= N
    """
    return tfidf
import math
from collections import defaultdict, Counter


import numpy as np
import math
from collections import Counter

def tf_idf_avec_son_dict(X, vocab):
    """
    Calcule les poids TF-IDF des mots présents dans 'vocab' à partir des messages X.
    Le score TF-IDF est cumulé sur tous les documents.

    Paramètres :
    ------------
    X : list[list[str]]
        Liste de messages, chaque message étant une liste de mots.

    vocab : set[str] ou list[str]
        Ensemble ou liste des mots à prendre en compte (filtrage du vocabulaire).

    Retour :
    --------
    dict[str, float] :
        Dictionnaire associant à chaque mot de 'vocab' son score TF-IDF cumulé.
    """
    index_mots = {mot: i for i, mot in enumerate(vocab)}
    tfidf = np.zeros((len(X),len(vocab)))
    df = np.zeros(len(vocab))  # document frequency
    N = len(X)

    # 1. Compter DF (Document Frequency)
    for doc in X:
        for mot in set(doc):
            if mot in index_mots:
                df[index_mots[mot]] += 1

    print("taille de TF : ", len(df))
    # 2. Calculer TF-IDF
    for i, message in enumerate(X):
        compteur = Counter(message) # compter la fréquence de chaque mot dans un message
        msg_len = len(message) # calcul la taille du message
        for mot, freq in compteur.items(): 
            if mot in index_mots:
                j = index_mots[mot]
                tf = freq / msg_len
                idf = math.log(N / (df[j] + 1))
                tfidf[i][j] += tf * idf
        i += 1
    # 3. Convertir en dictionnaire
    # calcul somme pour chaque mot pour former qu'un vecteur de 2 000 mots
    #res = np.zeros(len(vocab))
    #for j in range(len(vocab)):
    #    for i in range(len(X)):
    #        res[j] += tfidf[i][j]
    
    # res = tfidf.sum(axis=0)
    #print("taille du res : ",len(res) , " et de type : " , type(res))
    return tfidf


def tf_idf_par_groupe(groupes):
    """
    Calcule le TF-IDF moyen pour chaque mot dans chaque groupe, 
    en réutilisant la fonction tf_idf() définie précédemment.

    Paramètre :
    -----------
    groupes : dict[int, list[list[str]]]
        Dictionnaire : groupe -> liste de documents (liste de mots).

    Retour :
    --------
    dict[int, dict[str, float]] :
        Groupe -> {mot : score TF-IDF moyen}
    """
    tfidf_groupes = {}
    for numero, groupe in groupes.items():
        tfidf_groupes[numero] = tf_idf(groupe)  # appel à la fonction générique tf_idf
    return tfidf_groupes


def top_mots_par_groupe(tfidf_groupes, k=100):
    """
    Récupère les k mots avec les plus hauts scores TF_IDF pour chaque groupe

    Paramètre :
    -----------
    tfidf_groupes : dict[int, dict[str, float]]
        Dictionnaire des scores TF-IDF par groupe.

    k : int 
        Nombre de mots à extraire par groupe
    
    Retour : 
    --------
    dict[int, list[tuple[str, float]]] 
        groupe -> list triée de (mot, score)
    """
    def get_score(item):
        return item[1]
    
    top_mots = {}
    for g, scores in tfidf_groupes.items():
        top_mots[g] = sorted(scores.items(), key = get_score, reverse=True)[:k]
        #top_mots[g] = sorted(scores.items(), key = lambda x: -x[1])[:k]
    return top_mots


def afficher_top_mots(top_mots):
    """
    Affiche proprement les mots les plus caractéristiques de chaque groupe.
    """
    for g in sorted(top_mots.keys()):
        print(f"\nGroupe {g} :")
        for mot, score in top_mots[g]:
            print(f"  {mot:<20} {round(score, 3)}")


def vocabulaire_top_mot(dictionnaire):
    """
    Extrait le vocabulaire global (ensemble des mots uniques) à partir d'un dictionnaire 
    regroupant des documents par groupe.

    Paramètre :
    -----------
    dictionnaire : dict[int, [list (str, float)]]
        Un dictionnaire où chaque clé est un numéro de groupe, et chaque valeur 
        est une liste de documents, chaque document étant une liste de mots.

    Retour :
    --------
    set[str] :
        Ensemble contenant tous les mots uniques présents dans tous les groupes.
    """
    vocab = set()
    for _, docs in dictionnaire.items():
            vocab.update(set(docs))  # ajoute tous les mots du message
    return vocab
