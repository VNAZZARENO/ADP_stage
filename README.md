Résultat de mon stage au sein de CDGD3 à Groupe ADP pour la plateforme Roissy Charles de Gaulle.
On m'a donné l'opportunité de réaliser un meilleur outil que l'outil précédent réalisant les prévisions d'affluences aux postes d'inspection filtrage. 
J'ai donc de moi même réalisé la recherche, le traitement, les pipelines, l'entraitement et finalement la production de cet outil.

Les librairies utilisées sont:
- PyKalman pour le traitement des données
- SKLearn pour les pipelines
- XGBoost pour le modèle
- Optuna pour l'optimisation des hyperparamètres


L'outil est en réalité constitué d'outil indépendants entre eux. 
Il en possède 5 à l'heure actuelle :
- Concat V2
- Prévisions PIF
- Export PIF
- Seuil PIF
- Prévisions Taux Remplissage
Nous allons dans les prochaines parties détailler les outils (fonctions/univers)

Concat V2
Il y a deux type de programme prévisionnel : 
- Prévisions d'activité Air France & partenaires
- Prévisionnel SARIAP
Ces deux programmes ne sont pas directement joignables, et il faut effectuer des transformations de données afin de les joindre.
=> Cet outil sert à joindre ces deux programmes, afin que par la suite un seul fichier soit injecté dans les outils de prévisions.

Prévisions PIFs
Les prévisions PIFs sont des rapports d'affluences envoyés toutes les semaines à CDG9 qui leur permettent ensuite de dimensionner les personnels affectés à la sécurité et à l'inspection filtrage des passagers de l'aéroport. 
=> Cet outil permet de réaliser des prévisions pour chaque post de PIF sur le programme concaténé (sortie de l'outil Concat V2)

Export PIF
Les prévisions PIF sont regroupé en colonnes afin de permettre une utilisation ultérieure (Power BI, Python etc…). Cependant CDG9 requiert une vue de l'axe temporel horizontal et sous Excel. 
=> Cet outil permet cette mise en forme, ainsi que des post-traitement des prévisions demandé par CDG9 (regroupement K CNT et K CTR par exemple…)
 
Seuils PIF
Afin de vérifier rapidement si les prévisions d'affluence sont compatibles avec les seuils théorique de débit de traitement passagers dans les postes de control, un outil de visualisation graphique a été développé.
=> Cet outil sert à vérifier sur la semaine suivante d'activité les seuils de débits vis-à-vis des prévisions d'affluence

Taux de Remplissage
Les taux de remplissage des avions ne sont fournis que par Air France et ses partenaires sur le périmètre EST de l'aéroport. 
Ainsi pour le périmètre OUEST nous ne disposons pas de données fiabilisées par les compagnies elles-mêmes.
=> Cet outil a pour vocation de corriger ces prévisions de passagers qui sont erronées, et de les injecter dans la base SARIAP. 
