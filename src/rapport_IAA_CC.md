(1) En préambule : nom, prénom de chaque auteur, nom de l’équipe, date. 
(2) Un résumé des solutions que vous avez adoptées (algo, hyper-paramètres, descripteurs d’images), en expliquant pourquoi ces choix au regard d’alternatives mises en œuvre (ou pas), et les actions/mesures qui vous ont guidés vers ce choix. 
(3) Le rôle de chacun dans ce travail, le mode d’organisation, et les outils collaboratifs utilisés le cas échéant.
(4) Les valeurs des trois erreurs classiques considérées : les deux qui ont permis de choisir le modèle (erreur empirique, erreur réelle estimée (validation croisée ou TTS)), et celle en test obtenue sur les nouvelles données disponibles en début de séance; un commentaire expliquant les différences entre ces trois mesures. 
(5) Actions futures : quels seront vos développements à venir pour apprendre un meilleur modèle?

Enzo Nicaise, Tarun Rakesh, Gaétan Elouard-Bucchini, Antoine Tomasi (Equipe KING_BE4RN2000)

Test error: 0.2581, Train error: 0.1796 for hyperparameters: {'C': 2, 'kernel': 'linear', 'gamma': 'auto', 'degree': 3, 'class_weight': 'balanced'}

Test error: 0.3387, Train error: 0.3042 for hyperparameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 7, 'max_features': None}
Test error: 0.3790, Train error: 0.1326 for hyperparameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 10, 'max_features': 'sqrt'}

Test error: 0.2661, Train error: 0.1101 for hyperparameters: {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt'}