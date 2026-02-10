Smart Irrigation System using Kafka, Spark & Edge AI
Optimisation de l’irrigation en Mauritanie
Étudiant

Nom : Meimoune Sambe
N° : C18615

Université

Faculté des Sciences et Techniques – UNA

Encadré par

Dr Mohamed Mahmoud El Benany

📌 Description du projet

Ce projet présente un système intelligent d’irrigation basé sur :

Apache Kafka (collecte de données en temps réel)

Apache Spark Streaming (traitement des données)

Machine Learning (prédiction des besoins en eau)

Edge Computing (décision locale)

Streamlit (visualisation)

Le système analyse :

Température

Humidité du sol

Conditions climatiques

Pour recommander automatiquement l’irrigation optimale.

Contexte mauritanien

Ce projet vise à aider l’agriculture dans des zones comme :

Rosso

Kaédi

Boghé

En réduisant :

Le gaspillage d’eau

Les coûts agricoles

Les pertes de production

📊 Sources de données agricoles

Références utilisées :

FAO : https://www.fao.org/faostat/

World Bank Data : https://data.worldbank.org

Climate Data : https://www.climate-data.org

🛠 Technologies utilisées

Python

Apache Kafka

Apache Spark

Machine Learning (Scikit-learn)

Streamlit

Docker (optionnel)

📂 Structure du projet
irrigation-project/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── irrigation_model.pkl
│
├── kafka_producer.py
├── spark_streaming.py
├── train_model.py
├── edge_inference.py
├── dashboard.py
└── README.md

⚙️ Installation
1️⃣ Installer les dépendances Python
pip install pandas numpy scikit-learn kafka-python pyspark streamlit

▶️ MÉTHODE COMPLÈTE POUR LANCER LE PROJET

Suivre cet ordre exactement :

🔹 Étape 1 — Démarrer Zookeeper

Dans le dossier Kafka :

bin/zookeeper-server-start.sh config/zookeeper.properties


(Sur Windows)

.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

🔹 Étape 2 — Démarrer Kafka
bin/kafka-server-start.sh config/server.properties


(Sur Windows)

.\bin\windows\kafka-server-start.bat .\config\server.properties

🔹 Étape 3 — Lancer le Producer (simulation des capteurs)

Dans le dossier du projet :

python kafka_producer.py


Ce script envoie :

Température

Humidité

Données agricoles

vers Kafka.

🔹 Étape 4 — Lancer Spark Streaming
spark-submit spark_streaming.py


Spark va :

Lire les données depuis Kafka

Nettoyer les données

Analyser les anomalies

Sauvegarder les résultats

🔹 Étape 5 — Entraîner le modèle (Machine Learning)
python train_model.py


Ce script :

Charge le dataset agricole

Entraîne le modèle IA

Sauvegarde le modèle dans :

models/irrigation_model.pkl

🔹 Étape 6 — Lancer le modèle en mode Edge (prédiction locale)
python edge_inference.py


Ce script :

Charge le modèle entraîné

Analyse les nouvelles données

Prédit les besoins en irrigation en temps réel

🔹 Étape 7 — Lancer le Dashboard
streamlit run dashboard.py


Ouvrir ensuite dans le navigateur :

http://localhost:8501


Le dashboard affiche :

Graphiques temps réel

Niveau d’humidité

Alertes d’irrigation

Prédictions IA

Architecture du système

Capteurs → Kafka → Spark → Modèle ML → Edge AI → Dashboard

Objectifs du projet

Optimiser l’irrigation intelligente

Réduire la consommation d’eau

Aider les agriculteurs mauritaniens

Utiliser l’IA en agriculture

📄 Article scientifique

L’article complet du projet est inclus dans ce dépôt GitHub.

Remerciements

Je tiens à remercier sincèrement :

Dr Mohamed Mahmoud El Benany
Pour son encadrement, son soutien et ses précieux conseils.

Auteur

Meimoune Sambe
Étudiant – Faculté des Sciences et Techniques UNA