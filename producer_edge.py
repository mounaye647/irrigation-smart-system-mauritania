"""
Edge Node - Producteur Kafka avec Prédiction Locale
Ce script simule un capteur IoT qui:
1. Collecte les données des capteurs
2. Fait une prédiction LOCALE (Edge Computing)
3. Envoie les données enrichies vers Kafka
"""

import pandas as pd
from kafka import KafkaProducer
import json
import time
from datetime import datetime
import joblib
import os

class EdgeIoTNode:
    """
    Nœud Edge IoT avec capacité de prédiction locale
    """
    
    def __init__(self, zone_name, kafka_broker='localhost:9092'):
        self.zone_name = zone_name
        self.kafka_broker = kafka_broker
        
        # Connexion Kafka
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Charger le modèle Edge local
        self.model = self.load_edge_model()
        
        print(f"🌐 Edge Node initialisé pour la zone: {zone_name}")
        print(f"   Broker Kafka: {kafka_broker}")
        print(f"   Modèle Edge: {'✅ Chargé' if self.model else '❌ Non disponible'}")
    
    def load_edge_model(self):
        """Charger le modèle Edge pré-entraîné pour cette zone"""
        model_path = f'/home/claude/models/edge_model_{self.zone_name.lower()}.pkl'
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                print(f"   📥 Modèle chargé: {model_path}")
                return model_data
            except Exception as e:
                print(f"   ⚠️  Erreur chargement modèle: {e}")
                return None
        else:
            print(f"   ⚠️  Modèle non trouvé: {model_path}")
            return None
    
    def predict_irrigation(self, sensor_data):
        """
        Prédiction locale du temps d'irrigation
        
        Args:
            sensor_data: dict avec humidity, temperature, ph, evapotranspiration
        
        Returns:
            dict avec prédiction enrichie
        """
        if self.model is None:
            return {
                'irrigation_time_predicted': None,
                'urgence': 'UNKNOWN',
                'edge_prediction': False
            }
        
        # Préparer les features
        feature_names = ['humidity', 'temperature', 'ph', 'evapotranspiration']
        df = pd.DataFrame([sensor_data])
        df = df[feature_names]
        
        # Prédiction
        irrigation_time = self.model['model'].predict(df)[0]
        
        # Déterminer urgence
        if irrigation_time > 15:
            urgence = "CRITIQUE"
        elif irrigation_time > 10:
            urgence = "ÉLEVÉE"
        elif irrigation_time > 7:
            urgence = "MOYENNE"
        else:
            urgence = "FAIBLE"
        
        return {
            'irrigation_time_predicted': round(float(irrigation_time), 2),
            'urgence': urgence,
            'edge_prediction': True,
            'model_r2': self.model['metrics']['r2']
        }
    
    def send_data(self, sensor_reading):
        """
        Traitement Edge + Envoi vers Kafka
        
        Args:
            sensor_reading: dict avec les lectures des capteurs
        """
        # 1. Prédiction locale (Edge Computing)
        prediction = self.predict_irrigation(sensor_reading)
        
        # 2. Enrichir les données
        message = {
            'zone': self.zone_name,
            'timestamp': datetime.now().isoformat(),
            'sensor_data': sensor_reading,
            'edge_prediction': prediction,
            'node_id': f'edge_node_{self.zone_name.lower()}'
        }
        
        # 3. Envoi vers Kafka (vers Fog/Cloud)
        try:
            self.producer.send('irrigation-edge-data', value=message)
            
            # Log
            urgence_emoji = {
                'CRITIQUE': '🔴',
                'ÉLEVÉE': '🟠',
                'MOYENNE': '🟡',
                'FAIBLE': '🟢',
                'UNKNOWN': '⚪'
            }
            
            print(f"\n📡 {urgence_emoji.get(prediction['urgence'], '⚪')} [{self.zone_name}] Données envoyées")
            print(f"   Humidité: {sensor_reading['humidity']}%")
            print(f"   Température: {sensor_reading['temperature']}°C")
            if prediction['edge_prediction']:
                print(f"   ⚡ Prédiction Edge: {prediction['irrigation_time_predicted']} min")
                print(f"   📊 Urgence: {prediction['urgence']}")
            
        except Exception as e:
            print(f"❌ Erreur envoi Kafka: {e}")
    
    def close(self):
        """Fermer la connexion Kafka"""
        self.producer.close()
        print(f"\n🔌 Edge Node {self.zone_name} déconnecté")


def simulate_edge_nodes(dataset_path, delay_seconds=3):
    """
    Simuler plusieurs Edge Nodes envoyant des données en parallèle
    
    Args:
        dataset_path: chemin vers le CSV
        delay_seconds: délai entre chaque envoi
    """
    df = pd.read_csv(dataset_path)
    
    # Créer un Edge Node par zone
    zones = df['zone'].unique()
    edge_nodes = {
        zone: EdgeIoTNode(zone_name=zone)
        for zone in zones
    }
    
    print(f"\n🚀 Démarrage de la simulation Edge Computing")
    print(f"   {len(edge_nodes)} Edge Nodes actifs")
    print(f"   Délai entre envois: {delay_seconds}s")
    print("="*70)
    
    try:
        # Envoyer les données de façon cyclique
        row_index = 0
        
        while row_index < len(df):
            row = df.iloc[row_index]
            zone = row['zone']
            
            # Préparer les données capteur
            sensor_data = {
                'humidity': float(row['humidity']),
                'temperature': float(row['temperature']),
                'ph': float(row['ph']),
                'evapotranspiration': float(row['evapotranspiration']),
                'irrigation_time_actual': float(row['recommended_irrigation_time_min'])
            }
            
            # Envoyer via l'Edge Node correspondant
            edge_nodes[zone].send_data(sensor_data)
            
            time.sleep(delay_seconds)
            row_index += 1
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Simulation interrompue par l'utilisateur")
    
    finally:
        # Fermer tous les Edge Nodes
        for node in edge_nodes.values():
            node.close()


if __name__ == "__main__":
    import sys
    
    # Vérifier que les modèles sont entraînés
    if not os.path.exists('/home/claude/models'):
        print("❌ Les modèles Edge ne sont pas entraînés!")
        print("   Exécutez d'abord: python edge_model.py")
        sys.exit(1)
    
    # Lancer la simulation
    simulate_edge_nodes(
        dataset_path='irrigation_dataset_mauritania.csv',
        delay_seconds=2  # 2 secondes entre chaque envoi
    )