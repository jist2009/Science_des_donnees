# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Import de packages externes
import numpy as np
import pandas as pd
import copy
import psutil

# ---------------------------


# ------------------------ A COMPLETER :
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """    
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        """ version 1
        predictions = np.array([self.predict(x) for x in desc_set])
        correct_predictions = np.sum(predictions == label_set)
        accuracy = correct_predictions / len(label_set)
        return accuracy
        """
        predictions  = np.array([self.predict(x) for x in desc_set])
        # paramètre : fonction, 0 : par colonne / 1 : par ligne, liste
        # la fonction apply_along_axis applique la fonction 
        return np.mean(predictions == label_set)


from scipy.spatial.distance import cdist 
from collections import Counter


class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        if k <= 0:
            raise ValueError("k doit être un entier positif.")
        
        Classifier.__init__(self, input_dimension)
        self.k = k  # Nombre de voisins
        self.X_train = None  # Exemples d'entraînement
        self.Y_train = None  # Labels d'entraînement            

        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.linalg.norm(self.X_train - x, axis=1)
        indices_proches = np.argpartition(distances, self.k)[:self.k]
        labels_proches = self.Y_train[indices_proches]
        return np.sum(labels_proches == 1) / self.k  # Évite une comparaison inutile
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return 1 if self.score(x) >= 0 else -1 
        
    def predict_many_knn(Xtrain, Ytrain, Xtest, k=5, metric='cosine'):
        D = cdist(Xtest, Xtrain, metric=metric)
        Ypred = []

        for i in range(D.shape[0]):
            voisins = np.argpartition(D[i], k)[:k]
            classes_voisines = Ytrain[voisins]
            prediction = Counter(classes_voisines).most_common(1)[0][0]
            Ypred.append(prediction)
        return np.array[Ypred]

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.X_train = desc_set
        self.Y_train = label_set
        
        

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self,input_dimension)
        self.learning_rate= learning_rate
        self.desc = None
        self.label = None
        if(init):
            self.w=np.zeros(input_dimension)
        else:
            self.w= np.zeros(input_dimension)
            for i in range(input_dimension):
                v= np.random.uniform(0,1)
                v = 2*v - 1
                v = v*0.001
                self.w[i]=v
        self.allw = [self.w.copy()]
            
        
        
    def get_allw(self):
        """ Récupère la liste des variations des poids du modèle. """
        return self.allw
    	
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        indices = [i for i in range(len(desc_set))]
        np.random.shuffle(indices)
        for i in indices:
            desc = desc_set[i]
            label_predict = self.predict(desc)
            if label_predict != label_set[i]:
                self.w += self.learning_rate * label_set[i] * desc # Mise à jour des poids
                #self.allw.append(self.w.copy())
    
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        liste_norme = []
        for i in range(nb_max):
            w_precedent = self.w.copy() # Sauvegarde des poids avant mise à jour
            
            self.train_step(desc_set, label_set)  # Mise à jour des poids
            self.allw.append(self.w.copy())
            # Calcul de la norme de la différence
            norme_diff = np.linalg.norm(abs(self.w - w_precedent))
            liste_norme.append(norme_diff)
            
            # Vérification de la convergence
            if abs(norme_diff) <= seuil:
                break  # Convergence atteinte
        return liste_norme
        
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
            
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
       

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        super().__init__(input_dimension, learning_rate, init)
       
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        indices = np.random.permutation(len(desc_set))
        for i in indices : 
            x_i = desc_set[i]
            y_i = label_set[i]
            f_xi = self.score(x_i)
            if f_xi * y_i < 1 : # condition d'erreur 
                self.w += self.learning_rate * (y_i - f_xi) * x_i
                #self.allw.append(self.w.copy()) # Sauvegarde des poids après mise à jour

    # je re écris la fonction predict pcq j'ai peur que lorsqu'il utilise super.predict(self, x), il utilise 
    # donc polymorphisme, la fonction super.predict appel score de la classe fille donc c ok
    # donc pas besoin d'initialiser la fonction predict 



class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        self.cl_bin = cl_bin # Classifieur binaire de référence
        self.classifiers = [] # Liste des classifieurs binaires
        self.classes = None 
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.classes = np.unique(label_set)
        self.classifiers = []

        for c in self.classes:
            print("créer copy")
            # Création du classifieur binaire pour la classe c
            cl = copy.deepcopy(self.cl_bin)

            # Création des labels temporaires : +1 pour c, -1 pour les autres
            ytmp = np.where(label_set == c, 1, -1)

            # Entrainement du classifieur sur ces labels modifiés
            cl.train(desc_set, ytmp)
            self.classifiers.append(cl)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return [cl.score(x) for cl in self.classifiers]
                
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return self.classes[np.argmax(scores)] # Sélectionne la meilleur des classes
"""
    def predict_many(self, X):
        all_scores= []

        for cl in self.classifiers:
            scores = np.apply_along_axis(cl.score, 1, X)
            all_scores.append(scores) # shape (n_classifiers, n_samples)

        all_scores = np.array(all_scores) # shape: (n_classes, n_samples)

        predictions = np.argmax(all_scores, axis=0)
        return self.classes[predictions]
"""

import math    
from collections import defaultdict, Counter

class ClassifierNaiveBayes(Classifier):
    def __init__(self):
        self.classes = None
        self.vocab = set()
        self.class_word_counts = {}
        self.class_doc_counts = {}
        self.total_docs = 0
        self.class_priors = {}
        self.word_probs = {}

    def train(self, X, Y):
        """
        X : list[list[str]] : chaque document est une liste de mots
        Y : list[int] : classes associées aux documents
        """
        self.classes = np.unique(Y)
        self.total_docs = len(X)

        # Initialisation
        self.class_doc_counts = Counter(Y)
        self.class_word_counts = {c: Counter() for c in self.classes}

        for x, y in zip(X, Y):
            self.class_word_counts[y].update(x)
            self.vocab.update(x)

        V = len(self.vocab)
        self.word_probs = {}

        # Calcul des probabilités avec lissage de Laplace
        for c in self.classes:
            total_words_in_class = sum(self.class_word_counts[c].values())
            self.word_probs[c] = {}

            for word in self.vocab:
                count = self.class_word_counts[c][word]
                self.word_probs[c][word] = (count + 1) / (total_words_in_class + V)

            # Prior de la classe
            self.class_priors[c] = self.class_doc_counts[c] / self.total_docs

    def predict(self, x):
        """
        x : list[str] : un document à classer
        Retourne : la classe avec la probabilité la plus élevée
        """
        scores = {}
        for c in self.classes:
            log_prob = np.log(self.class_priors[c])
            for word in x:
                if word in self.vocab:
                    log_prob += np.log(self.word_probs[c].get(word, 1 / (sum(self.class_word_counts[c].values()) + len(self.vocab))))
            scores[c] = log_prob

        return max(scores, key=scores.get)






















######################################################################################
######################################################################################
######################################################################################



def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    if len(P) in {0, 1}:
        return 0
    res = 0
    base_k = len(P)
    for proba in P:
        if proba > 0:
            res -= proba * math.log(proba, base_k)
    return res
    
def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    label, uniter = np.unique(Y, return_counts=True)
    uniter = uniter/sum(uniter)
    return shannon(uniter)

def entropie_conditionnelle(Xj, Y):
    """ Calcule l'entropie conditionnelle H(Y| Xj)"""
    valeurs, counts = np.unique(Xj, return_counts=True)
    entropie_cond = 0.0
    for v, count in zip(valeurs, counts):
        Y_v = Y[Xj == v]
        entropie_cond += (count/ len(Xj)) * entropie(Y_v)
    return entropie_cond

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    label, valeur = np.unique(Y, return_counts=True)
    indice_max = np.argmax(valeur)
    return label[indice_max]

import graphviz as gv

# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple 
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # test si exemple est dans self.les_fils
            # ok, on prend une catégorie 
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return None
    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.Les_fils:
            total += self.Les_fils[noeud].compte_feuilles()
        return total
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


def construit_AD(X, Y, epsilon, LNoms=[], profondeur=0, max_depth=10):
    """ Construit un arbre de décision récursivement.
        - epsilon : seuil de gain d'information pour arrêter.
        - LNoms : noms des colonnes.
    """
    # Cas 1 : profondeur maximale atteinte
    if profondeur >= max_depth:
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        return noeud

    entropie_ens = entropie(Y)

    # Cas 2 : pureté ou entropie faible
    if entropie_ens <= epsilon or len(np.unique(Y)) == 1:
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        return noeud

    # Recherche du meilleur attribut
    min_entropie = float('inf')
    i_best = -1
    Xbest_valeurs = None

    for i in range(X.shape[1]):
        entropie_cond = entropie_conditionnelle(X[:, i], Y)
        if entropie_cond < min_entropie:
            min_entropie = entropie_cond
            i_best = i
            Xbest_valeurs = np.unique(X[:, i])

    # Cas 3 : aucun split utile trouvé
    if i_best == -1 or entropie_ens - min_entropie < epsilon:
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        return noeud

    # Création du nœud
    if LNoms:
        noeud = NoeudCategoriel(i_best, LNoms[i_best])
    else:
        noeud = NoeudCategoriel(i_best)

    # 🔁 Création des fils
    for v in Xbest_valeurs:
        mask = X[:, i_best] == v
        Xv, Yv = X[mask], Y[mask]
        if len(Yv) == 0:
            feuille = NoeudCategoriel(-1, "Label")
            feuille.ajoute_feuille(classe_majoritaire(Y))
            noeud.ajoute_fils(v, feuille)
        else:
            noeud.ajoute_fils(v, construit_AD(Xv, Yv, epsilon, LNoms, profondeur + 1, max_depth))

    return noeud


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)  # Appel du constructeur de la classe mère
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)
        
    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def draw(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)























##############################################################################
##############################################################################
##############################################################################


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)


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




class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        # si l'arbre est une feuille
        if self.est_feuille():
            return self.classe

        if exemple[self.attribut] > self.seuil : 
            return self.Les_fils["sup"].classifie(exemple)
        else : 
            return self.Les_fils["inf"].classifie(exemple)

        # si dans aucune des cases : 
        return 0
        
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1

        return self.Les_fils["sup"].compte_feuilles() + self.Les_fils["inf"].compte_feuilles() 
        
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g


def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = 0.0  # meilleur gain trouvé (initalisé à 0.0 => aucun gain)
        i_best = -1     # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_seuil = None
        Xbest_entropie = 100
        
        # parcours de tous les attributs 
        # les liste ne sont pas utiles
        for i in range(nb_col):
            (seuil, entropie), (liste_coupure, liste_entropie) = discretise(X, Y, i)
        
            if entropie < Xbest_entropie:
                i_best = i
                gain_max = entropie - Xbest_entropie
                Xbest_seuil = seuil
                Xbest_entropie = entropie
                """
                partie_inf_X = []
                partie_inf_Y = []
                partie_sup_X = []
                partie_sup_Y = []
                
                for k in range(nb_lig):
                    if X[k][i] <= seuil:
                        partie_inf_X.append(X[k])
                        partie_inf_Y.append(Y[k])
                    else:
                        partie_sup_X.append(X[k])
                        partie_sup_Y.append(Y[k])
                
                Xbest_tuple = ((np.array(partie_inf_X), np.array(partie_inf_Y)),
                               (np.array(partie_sup_X), np.array(partie_sup_Y)))
                """
                Xbest_tuple = ((X[X[:, i] <= seuil], Y[X[:, i] <= seuil]),
                               (X[X[:, i] > seuil], Y[X[:, i] > seuil]))
                
        if (i_best != -1): # Un attribut qui amène un gain d'information >0 a été trouvé
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud

class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
           # print(desc_set[i,:] ," ",  label_set[i] , " " ,self.predict(desc_set[i,:]))
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)

