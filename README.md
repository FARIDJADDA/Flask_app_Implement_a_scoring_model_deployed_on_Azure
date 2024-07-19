# Flask App: Implement a Scoring Model Deployed on Azure

## Description
Ce projet consiste à construire un modèle de scoring prédictif pour estimer la probabilité de faillite d'un client, puis à le déployer dans le cloud. Le projet est réalisé en suivant une approche MLOps de bout en bout, incluant le tracking des expérimentations, l'analyse du data drift en production, et la mise en place d'une API pour interagir avec le modèle.

## Missions

1. **Construire un modèle de scoring** qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2. **Analyser les features** qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale).
3. **Mettre en production le modèle de scoring** via une API et réaliser une interface de test de cette API.
4. **Mettre en œuvre une approche globale MLOps** incluant le tracking des expérimentations et l'analyse en production du data drift.

## Étapes du projet

### Partie 1 : Construction du modèle de scoring

#### Étape 1 : Préparer l'environnement d'expérimentation
- Initialiser un environnement MLFlow pour le tracking des modèles.
- Mettre en place une UI pour la visualisation et la comparaison des expérimentations, ainsi que le stockage centralisé des modèles.

#### Étape 2 : Préparer les données pour la modélisation
- Choisir un Kernel Kaggle pertinent et riche en feature engineering.
- Adapter le Kernel à l'environnement technique.

#### Étape 3 : Créer un score métier pour l'entraînement des modèles
- Définir un score via une pondération sur les faux positifs et faux négatifs.
- Définir une stratégie d'évaluation et d'optimisation des modèles en utilisant ce score métier.

#### Étape 4 : Simuler et comparer plusieurs modèles
- Tester plusieurs modèles via une GridSearchCV et les comparer à une baseline.
- Gérer le déséquilibre entre les classes si nécessaire.
- Analyser la feature importance globale et locale du meilleur modèle retenu.

### Partie 2 : Déploiement et mise en production

#### Étape 1 : Déployer l'API dans le cloud
- Utiliser Git pour le versionning du code de l'API.
- Mettre en production l'API via les fonctionnalités de GitHub Actions et en utilisant une solution Cloud.

#### Étape 2 : Mettre en place une interface de test de l'API pour simuler un scoring client
- Utiliser Streamlit pour lier l'interface au modèle via l'API.

#### Étape 3 : Vérifier le travail et préparer la soutenance
- Préparer la soutenance en mettant en avant le processus de modélisation, les décisions prises, et l'importance des features.
- Anticiper les questions sur les méthodes et résultats pour une interaction constructive lors de la discussion.
- Prévoir des questions ciblées pour le débriefing afin d'obtenir des retours utiles pour la progression.

## Structure du Projet

```
PROJET_7
│
├── api
│ ├── predict_model.py
│ └── test_local_api_v2.ipynb
│
├── data
│ ├── processed
│ └── raw
│
├── mlruns
│
├── models
│
├── notebooks
│
├── output
│
├── test
│
├── app.py
│
├── requirements.txt
│
└── .gitignore
```


## Installation

1. **Cloner le dépôt** :
```bash
   git clone https://github.com/yourusername/Flask_app_Implement_a_scoring_model_deployed_on_Azure.git
   cd Flask_app_Implement_a_scoring_model_deployed_on_Azure
```

2. **Installer les dépendances :** :
```bash
    pip install -r api/requirements.txt
```

3. **Configurer MLFlow :** :
Suivre les instructions de configuration de MLFlow pour votre environnement.

## Utilisation :

1. **Lancer l'API localement :**

```bash
    python api/app.py
```

2. **Tester l'API :**

Utiliser le notebook test_local_api_v2.ipynb pour envoyer des requêtes à l'API et vérifier les prédictions.

3. **Déployer l'API dans le cloud :**

Utiliser **GitHub Actions** et une solution cloud comme Azure pour mettre en production l'API.
Utiliser l'interface de test :

Lancer l'application Streamlit pour interagir avec l'API et tester le scoring client.

# Contribution
Les contributions sont les bienvenues. Merci de bien vouloir ouvrir une issue pour discuter des changements majeurs avant de soumettre une pull request.
