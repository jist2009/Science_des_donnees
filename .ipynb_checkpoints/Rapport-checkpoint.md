# Projet SDD

**Sujet de Stéphane Ji** : Peut on regrouper automatiquement des documents selon leur contenu et identifier si ces groupes reflètent des catégories distinctes ?

## Sommaire  

### 1. Introduction
- Contexte et problématique
- Objectifs du projet

### 2. Prétraitement des données 
- Netttoyage et normalisation du texte
- Suppression des stop words, lemmatisation
- Vectorisation (One hot binary, Bag of Words, TF-IDF)

### 3. Analyse exploratoire
- Visualisation des données (PCA, t-SNE)
- Répartition des documents par catégories
- Réflexion sur la catégorisation naturelle

### 4. Classification supervisée
- Méthodes utilisées :
  - K-Nearest Neighbors (KNN)
  - Perceptron (avec et sans biais)
  - Arbre de décision
  - Arbre numérique
  - Naive Bayes
- Evaluation :
  - Cross-validation
  - Matrices de confusion
  - Scores de précision, rappel, F1

### 5. Classification non supervisée
- Méthodes de clustering :
  - K-moyennes
  - K-médoïdes
  - Clustering hiérarchique :
    - Linkage simple
    - Linkage complet
    - Linkage moyen
    - Linkage centroid
- Evaluation :
  - Dendrogrammes
  - Indices (Silhouette, Davies-Bouldin)

### 6. Comparaison et interprétation 
- Comparaison des performances supervisées vs non supervisées
- Analyse des regroupements : reflètent-ils les catégories initiales ?
- Limites rencontrées

### 7. Conclusion 
- Résumé des résultats
- Ouvertures possibles

### Annexes 
- Code source
- Jeux de données utilisés
- Références



## Introduction 

Aujourd'hui, la quantité de documents textuels disponibles en ligne et dans les bases de données ne cesse de croître. Il devient alors essentiel de disposer d’outils automatiques capables d’analyser, de regrouper et de classer ces documents selon leur contenu. L’objectif de ce projet est d’explorer des méthodes d’analyse de données textuelles permettant de détecter des regroupements naturels de documents, et de déterminer si ces regroupements correspondent à des catégories thématiques distinctes.

Pour cela, nous avons étudié deux grands axes de l’apprentissage automatique : l’apprentissage supervisé, où les documents sont étiquetés par des catégories connues, et l’apprentissage non supervisé, où l’algorithme doit identifier les groupes sans information préalable. Nous avons appliqué différentes techniques de classification (KNN, perceptron, arbre de décision, Naive Bayes) et de clustering (k-moyennes, k-médoïdes, clustering hiérarchique avec plusieurs critères de linkage), après un prétraitement approfondi des textes et une vectorisation adaptée (Bag-of-Words, TF-IDF).

L’analyse comparative de ces méthodes nous permettra d’évaluer leur capacité à reconstruire des catégories pertinentes et de réfléchir aux limites de ces approches sur des données textuelles complexes.


## 2. Prétraitement des données

Le projet s’appuie sur deux bases de données fournies :
- **20newsgroups** : un ensemble de documents textuels répartis en 20 catégories thématiques. Chaque message est accompagné d’un identifiant de groupe (un entier entre 0 et 19).
- **stopwords** : une liste de mots fréquents (comme "le", "et", "est", etc.) qui n’apportent généralement aucune information discriminante. Leur élimination permet de réduire le bruit lors de l’analyse.

L’objectif de cette étape est de transformer ces textes bruts en représentations numériques exploitables par les algorithmes d’apprentissage automatique. Pour cela, plusieurs traitements ont été réalisés.

### Nettoyage et filtrage lexical

Chaque message a d'abord été nettoyé en supprimant :
- la ponctuation,
- les majuscules (passage en minuscules),
- les caractères non alphabétiques.

Ensuite, un **filtrage par les stopwords** a été appliqué pour retirer les mots vides de sens catégoriel. Le projet propose de comparer les performances entre un jeu de données filtré et non filtré. Par souci de temps et en se basant sur un raisonnement théorique, seule la version **filtrée** a été utilisée ici, avec l’hypothèse qu’elle offrirait de meilleures performances de classification.

### Construction du dictionnaire

Un premier essai consistant à créer un dictionnaire global à partir de tous les mots des messages a généré un vocabulaire de plus de **150 000 mots**, ce qui s’est révélé inexplorable sur un ordinateur personnel en raison de la consommation excessive de mémoire (vectorisation trop lourde, kernel interrompu).

Pour résoudre ce problème, une stratégie de **réduction du vocabulaire** a été mise en place :
1. Les messages ont été regroupés par **groupe d’appartenance** (de 0 à 19).
2. Pour chaque groupe, un **comptage des fréquences** (ou une normalisation via TF-IDF) a permis d’identifier les **100 mots les plus caractéristiques** de ce groupe.
3. L’union de ces listes a permis de construire un **dictionnaire réduit de 2000 mots** (100 mots x 20 groupes), représentatif du contenu de chaque catégorie.

Ce choix repose sur l’hypothèse centrale du projet : **chaque groupe possède un vocabulaire spécifique** qui permet d’identifier ses documents. Ainsi, en se concentrant uniquement sur les mots les plus discriminants, on réduit la dimensionnalité tout en conservant l’essentiel de l’information catégorielle.

### Vectorisation des documents

Trois méthodes de vectorisation ont été envisagées :
- **One-hot binary** : chaque mot présent dans le document est codé par un 1, sinon 0.
- **Bag of Words** : codage basé sur les fréquences d’apparition des mots.
- **TF-IDF (Term Frequency – Inverse Document Frequency)** : pondération qui favorise les mots fréquents dans un document mais rares dans l’ensemble du corpus.

Les représentations TF-IDF ont été principalement utilisées pour leur capacité à mieux refléter l’importance relative des mots dans la classification des documents.

---

Ce prétraitement joue un rôle central dans le succès des méthodes d’apprentissage utilisées dans la suite du projet. Une représentation bien construite du texte conditionne fortement la capacité des algorithmes à détecter des regroupements pertinents ou à apprendre efficacement une classification.



## 3. Analyse exploratrice 

### Visualisation PCA 

La projection des vecteurs TF-IDF des mots en deux dimensions via PCA permet de visualiser la structure globale du vocabulaire en lien avec les groupes de documents.

Analyse du 1er graphe (plus large dispersion)
Observations :
- Forte dispersion sur l’axe des composantes principales, notamment Composante 1.
- Les groupes semblent plus éclatés et séparés dans l’espace.
- Plusieurs mots forment des alignements verticaux ou diagonaux : cela peut être dû à des mots caractéristiques fortement corrélés à un seul groupe.

Interprétation :
- Ce graphe conserve une part importante de la variance totale (probablement avec les 2 premières composantes).
- Il suggère que certains groupes ont un vocabulaire très spécifique (fortement discriminé par TF-IDF).
- Par contre, quelques zones de recouvrement montrent que certains groupes partagent un vocabulaire partiel.


Analyse du 2e graphe (plus compact, moins dispersé)
Observations :
- Les points sont plus tassés vers la gauche, particulièrement sur Composante 1.
- Les nuages de mots sont plus structurés horizontalement, en bandes.

Interprétation :
- Tu as probablement filtré davantage les mots, ou utilisé un vocabulaire réduit (comme uniquement les mots uniques).
- Cela conduit à un noyau de vocabulaire partagé plus restreint, et donc à moins de variance globale.
- Les groupes sont moins bien séparés visuellement, ce qui peut indiquer que les mots conservés sont moins discriminants globalement.

Conclusion : 
- La concentration centrale correspond à des mots partagés entre plusieurs groupes, peu informatifs (ex : mots techniques ou neutres).
- Les extrémités de la projection révèlent les mots les plus utiles pour classifier les documents, car ils sont typiques d’un groupe et absents ailleurs.
- La structure diagonale et la séparation spatiale de certains sous-groupes appuient l'idée que le vocabulaire porte une information catégorielle exploitable.

### Visualisation t-SNE 

Projection t-SNE des mots selon leurs scores TF-IDF
Hypothèses (confirmées par affichage) :
    Chaque point représente un mot vectorisé par ses scores TF-IDF sur 20 groupes.
    La projection 2D a été faite avec t-SNE, pour révéler la structure locale des similarités entre mots.
    La couleur indique le groupe d’origine du mot (entre 0 et 19, encodés ici de 0 à 19 via color).
    Axes : Composante 1 (horizontal), Composante 2 (vertical).

Analyse de la structure :

1. Organisation en clusters compacts
    Les mots se regroupent en amas denses et séparés : excellent résultat pour t-SNE.
    Chaque couleur est clairement localisée, ce qui montre que :
        Les mots les plus caractéristiques de chaque groupe ont des signatures TF-IDF très différentes.
        Le vocabulaire est bien spécifique par classe, ce qui confirme la validité de ton traitement.

2. Groupes particulièrement bien séparés
    À gauche :
        Le groupe groupe 0 est très isolé, en haut à gauche. C’est le groupe le plus distinct.
    En bas :
        Un groupe rouge(9) et orange(10) est également fortement distinctif.
    Au centre-droit :
        Une zone de densité moyenne à forte, avec plusieurs classes regroupées, formant un noyau sémantique partagé.

3. Symétrie et orientation

    L’espace est non symétrique, ce qui est normal pour t-SNE.
    Il n’y a pas d’alignement diagonaux ou circulaires, ce qui indique une bonne captation des variations locales.

Interprétation :
Groupes très séparés spatialement	Mots très spécifiques à leur groupe — bien discriminés par TF-IDF
Groupes proches mais non confondus	Vocabulaire partiellement partagé (groupes similaires thématiquement)
Amas denses	Les mots ont des profils TF-IDF proches → forte cohérence sémantique locale
Outliers très à gauche/haut/bas	Mots rares mais très significatifs (probablement propres à un sujet unique)

Conclusion :
    Cette projection t-SNE est très satisfaisante : elle révèle des groupes lexicaux bien distincts, alignés avec les classes originales.
    Tu peux désormais utiliser cette visualisation pour :
        justifier la pertinence des TF-IDF pour caractériser les groupes
        expliquer l’intérêt d’un vocabulaire filtré pour la classification automatique
        explorer quels mots sont les plus discriminants (via hover ou centroides)





## 3. CLASSIFIEUR SUPERVISEE

### KNN 

temps : non calculé 
intervalle de k : { 1 à 30 par 5}
performance : 
[np.float64(0.05163428135620209),
 np.float64(0.05163428135620209),
 np.float64(0.05163428135620209),
 np.float64(0.05163428135620209),
 np.float64(0.05163428135620209),
 np.float64(0.05163428135620209)]

pas de matrice de confusion

INTERPRETATION : 
- KNN non efficace pour la classfication de cette donnée
ou
- Il doit y avoir des erreurs sur le classifieur ou la fonction validation croisé car il renvoie exactement 6 fois la meme valeur.   


### Perceptron 


Temps total de la validation croisé : 373.63 s
donc temps par itération : 35 s
taux moyen :  0.42826489605551527  taux ecart :  0.016405405514965676
matrice de confusion dans le dossier "image supervisee"

INTERPRETATION:

Des résultats meilleurs que KNN.
La matrice de confusion montre que les groupes 9 à 11 ont des taux de classification supérieur à 50 % tandis que les groupes 1, 3, 12 , 18 et 19 ont des taux de classification inférieur à 30%. je suppose que les groupes 1,,3,12,18 et 19 doivent avoir les mêmes familles de mots ou qu'ils se ressmeblent. 


### Arbre de décision 

Non terminé 

la première itération affiche un taux de classification de 25% pour un temps d'execution de 15 secondes

### Naive Bayes

Le test pour ce classificateur n'est pas exécuter par les erreurs et par manque de temps.




# Avertissement 

Je m’étais initialement trompé : je pensais que le dictionnaire contenait 2000 mots distincts, alors qu’en réalité il n’en contient qu’environ 800. Cela signifie qu’il y a environ 1200 doublons parmi les 2000 mots sélectionnés. Cette redondance affecte la représentativité du dictionnaire, et par conséquent, biaisent certaines analyses, même si certains groupes restent bien différenciés.

Par manque de temps, je poursuis néanmoins les expériences avec ce dictionnaire sans le reconstruire, bien que cela affecte légèrement la fiabilité des résultats.

# Suggestions d’améliorations

1. Il serait judicieux de prendre les k meilleurs mots par groupe (ex : 200 ou 500), plutôt qu’un total fixe de 100 mots.
En effet, des mots trop fréquents comme "would" sont présents dans plusieurs groupes et ne sont pas discriminants.
l’inverse, des mots moins fréquents mais plus spécifiques seraient davantage caractéristiques de leur groupe,bien qu’ils ne soient pas présents dans tous les messages.
Il faudrait donc trouver un équilibre entre fréquence (apparition suffisante) et spécificité (pertinence catégorielle).

2. Ajustement du nombre d’itérations pour la validation croisée
Je pourrais également faire varier le nombre de folds de la validation croisée, par exemple entre 3 et 20.
Cela permettrait d’observer le point optimal qui maximise le taux de bonne classification tout en minimisant le coût computationnel (temps, RAM, etc.).


# Remarque 

Au début, j'ai commencé le projet avec des taches que je n'utilise plus. Par exemple, le dictionnaire de tous les message était 150 000 mots et avec 20 000 messages cela tuait le kernel. De ce fait, pour ne pas réduire le jeu de donnée initiale, j'ai remarqué que chaque message n'utilisait que 100 mots et la vectorisation du message avec la taille du dictionnaire consommé beaucoup de RAM par rapport pour un petit message. Donc, j'avais créer un X de tous les messages et chaque message ne garder que les index de leurs mots. 
J'ai codé 4 fonctions qui vectorise les message en binary, comptage, normalisation et tfidf.
Mais après codage, j'ai abandonné cette idée car ce vecteur n'est pas un vecteur utilisable pour les classifieurs et les clusterings.

De plus, pour les analyse :
1. j'avais gardé que les 10 mots les plus caractéristiques du groupe que j'ai utilisés pour la présentation. 










