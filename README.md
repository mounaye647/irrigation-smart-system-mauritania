Smart Irrigation System using Kafka, Spark & Edge AI
Optimisation de l’irrigation en Mauritanie
Étudiant


Description du projet

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

Sources de données agricoles


Technologies utilisées

Python

Apache Kafka

Apache Spark

Machine Learning (Scikit-learn)

Streamlit

Docker (optionnel)


Installation
Installer les dépendances Python
pip install pandas numpy scikit-learn kafka-python pyspark streamlit

MÉTHODE COMPLÈTE POUR LANCER LE PROJET

Suivre cet ordre exactement :

Étape 1 — Démarrer Zookeeper

Dans le dossier Kafka :

bin/zookeeper-server-start.sh config/zookeeper.properties


(Sur Windows)

.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

Étape 2 — Démarrer Kafka
bin/kafka-server-start.sh config/server.properties


(Sur Windows)

.\bin\windows\kafka-server-start.bat .\config\server.properties

Étape 3 — Lancer le Producer (simulation des capteurs)

Dans le dossier du projet :

python kafka_producer.py

Étape 4 — Lancer Spark Streaming
spark-submit spark_streaming.py


Étape 5 — Entraîner le modèle (Machine Learning)
python train_model.py

Étape 6 — Lancer le modèle en mode Edge (prédiction locale)
python edge_inference.py

Étape 7 — Lancer le Dashboard
streamlit run dashboard.py
