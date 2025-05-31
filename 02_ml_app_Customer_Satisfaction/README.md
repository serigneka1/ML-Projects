# Customer Satisfaction ML App

Un projet de Machine Learning permettant de prédire la satisfaction des clients à partir de leurs interactions, données démographiques et historiques d'achat.

---

## Objectifs

- Prétraitement automatisé des données clients
- Entraînement d’un modèle de classification (satisfait / insatisfait)
- Validation et sauvegarde du modèle
- Pipeline reproductible avec Zenml
- Déploiement du modèle

---

## Structure du projet

```bash
02_ml_app_Customer_Satisfaction/
│
├── data/ # Données brutes et traitées (ignorées par git)
├── steps/ # Étapes individuelles du pipeline ZenML
├── pipelines/ # Définition du pipeline d'entraînement
├── run_pipeline.py # Script principal pour exécuter le pipeline
├── requirements.txt # Dépendances du projet
├── .gitignore # Fichiers exclus du versionnement
└── README.md # Ce fichier
````

---
##  Tech Stack

- **Langage** : Python 3.10 ( au plus 3.10 pour compatilité avevc zenflow)
- **Framework ML** : scikit-learn
- **Orchestration** : ZenML
- **Visualisation / Log** : ZenML Dashboard
- **Stockage des artefacts** : local ( plutard configurable vers cloud)

---
## Lancer le pipeline
1. **Créer un environnement virtuel :**

```bash
python -m venv .venv
source .venv/bin/activate  # ou .\.venv\Scripts\activate sur Windows
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```
3. **Lancer le pipeline :**
```bash
python run_pipeline.py
```
3. **Pour voir le dashboard de monitoring de la pipeline avec zenml :**
```bash
zenml login --local . #  si port non précisé alors le dashboard à visiter ici localhost:8237
zenml login --local --blocking . # pour windows ajouter --locking pour forcer le blockage
```

## Fonctionnalités (à venir)
Backend avec FastAPI
Frontend avec HTML/CSS/bootstrap
Déploiement du modèle
Versioning des modèles avec MLflow

---
## Contribuer
Les contributions sont les bienvenues ! Merci de proposer une issue ou une pull request.

## Licence
Ce projet est sous licence MIT.

## Auteur
Serigne - GitHub | kaserigne69@gmail.com - Linkedin | linkedin.com/in/serigneka
