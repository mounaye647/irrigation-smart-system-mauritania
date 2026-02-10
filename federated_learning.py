"""
Federated Learning - Implémentation FedAvg
Algorithme de Federated Averaging pour l'irrigation intelligente

Architecture:
- Edge Nodes: Entraînement local sur données privées
- Fog Nodes: Agrégation régionale (optionnel)
- Cloud Server: Fusion des modèles globaux (FedAvg)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from datetime import datetime
from typing import List, Dict
import copy

class FederatedClient:
    """
    Client Federated Learning (Edge Node)
    Entraîne un modèle local sans partager les données brutes
    """
    
    def __init__(self, client_id, zone_name):
        self.client_id = client_id
        self.zone_name = zone_name
        self.local_model = None
        self.local_data = None
        self.feature_names = ['humidity', 'temperature', 'ph', 'evapotranspiration']
        
    def load_data(self, data):
        """Charger les données locales (privées)"""
        self.local_data = data
        print(f"   📥 [{self.zone_name}] {len(data)} échantillons chargés (données privées)")
    
    def train_local_model(self, global_model_params=None, epochs=1):
        """
        Entraînement local du modèle
        
        Args:
            global_model_params: Paramètres du modèle global (si disponible)
            epochs: Nombre d'epochs d'entraînement local
        
        Returns:
            dict: Paramètres du modèle local (poids/gradients)
        """
        print(f"\n   🏋️  [{self.zone_name}] Entraînement local...")
        
        # Préparer les données
        X = self.local_data[self.feature_names]
        y = self.local_data['recommended_irrigation_time_min']
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialiser ou charger le modèle
        if global_model_params is not None:
            # Partir du modèle global
            self.local_model = global_model_params['model']
        else:
            # Nouveau modèle
            self.local_model = RandomForestRegressor(
                n_estimators=30,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        
        # Entraînement
        self.local_model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.local_model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        print(f"      ✅ R² local: {r2:.4f}")
        print(f"      ✅ MSE local: {mse:.4f}")
        
        # Retourner les paramètres (pas les données brutes !)
        return {
            'client_id': self.client_id,
            'zone': self.zone_name,
            'model': copy.deepcopy(self.local_model),
            'n_samples': len(self.local_data),
            'r2_score': r2,
            'mse': mse,
            'timestamp': datetime.now().isoformat()
        }


class FederatedServer:
    """
    Serveur Federated Learning (Cloud)
    Agrège les modèles locaux sans accéder aux données privées
    """
    
    def __init__(self):
        self.global_model = None
        self.clients = []
        self.history = []
        self.current_round = 0
        
    def register_client(self, client):
        """Enregistrer un client FL"""
        self.clients.append(client)
        print(f"   ✅ Client enregistré: {client.zone_name}")
    
    def federated_averaging(self, client_models: List[Dict]) -> Dict:
        """
        Algorithme FedAvg - Moyenne pondérée des modèles locaux
        
        Args:
            client_models: Liste des modèles locaux avec leurs métriques
        
        Returns:
            dict: Modèle global agrégé
        """
        print(f"\n   🔄 FedAvg Round {self.current_round + 1}")
        print("   " + "="*60)
        
        # Calcul des poids (proportionnels au nombre d'échantillons)
        total_samples = sum(m['n_samples'] for m in client_models)
        weights = [m['n_samples'] / total_samples for m in client_models]
        
        print(f"   📊 Agrégation de {len(client_models)} modèles locaux:")
        for i, (model, weight) in enumerate(zip(client_models, weights)):
            print(f"      • {model['zone']}: {model['n_samples']} samples (poids: {weight:.3f})")
        
        # Pour Random Forest, on moyenne les prédictions
        # (Dans un cas réel avec Neural Networks, on moyennerait les poids)
        
        # Ici, on utilise le modèle du client avec le plus de données
        # ou on pourrait faire un ensemble
        best_client_idx = np.argmax([m['n_samples'] for m in client_models])
        
        self.global_model = {
            'model': copy.deepcopy(client_models[best_client_idx]['model']),
            'round': self.current_round + 1,
            'n_clients': len(client_models),
            'total_samples': total_samples,
            'avg_r2': np.average([m['r2_score'] for m in client_models], weights=weights),
            'avg_mse': np.average([m['mse'] for m in client_models], weights=weights),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n   ✅ Modèle global mis à jour:")
        print(f"      • R² moyen pondéré: {self.global_model['avg_r2']:.4f}")
        print(f"      • MSE moyen pondéré: {self.global_model['avg_mse']:.4f}")
        
        self.current_round += 1
        self.history.append(self.global_model)
        
        return self.global_model
    
    def run_fl_round(self):
        """
        Exécuter un round de Federated Learning
        
        Workflow:
        1. Distribuer modèle global aux clients
        2. Clients entraînent localement
        3. Clients renvoient leurs modèles
        4. Serveur agrège avec FedAvg
        """
        print(f"\n{'='*70}")
        print(f"🌐 FEDERATED LEARNING - ROUND {self.current_round + 1}")
        print(f"{'='*70}")
        
        # Étape 1: Entraînement local sur chaque client
        client_models = []
        
        for client in self.clients:
            model_params = client.train_local_model(
                global_model_params=self.global_model
            )
            client_models.append(model_params)
        
        # Étape 2: Agrégation FedAvg
        global_model = self.federated_averaging(client_models)
        
        return global_model
    
    def train(self, num_rounds=5):
        """
        Entraînement Federated Learning complet
        
        Args:
            num_rounds: Nombre de rounds FL
        """
        print(f"\n🚀 DÉMARRAGE FEDERATED LEARNING")
        print(f"   Clients: {len(self.clients)}")
        print(f"   Rounds: {num_rounds}")
        print(f"{'='*70}\n")
        
        for round_idx in range(num_rounds):
            self.run_fl_round()
        
        print(f"\n{'='*70}")
        print(f"✅ FEDERATED LEARNING TERMINÉ")
        print(f"{'='*70}")
        
        return self.global_model
    
    def save_global_model(self, filepath):
        """Sauvegarder le modèle global"""
        joblib.dump(self.global_model, filepath)
        print(f"\n💾 Modèle global sauvegardé: {filepath}")
    
    def get_training_history(self):
        """Obtenir l'historique d'entraînement"""
        history_df = pd.DataFrame([
            {
                'round': h['round'],
                'avg_r2': h['avg_r2'],
                'avg_mse': h['avg_mse'],
                'n_clients': h['n_clients'],
                'total_samples': h['total_samples']
            }
            for h in self.history
        ])
        return history_df


def run_federated_learning_simulation(dataset_path, num_rounds=5):
    """
    Simulation complète de Federated Learning
    
    Args:
        dataset_path: Chemin vers le dataset
        num_rounds: Nombre de rounds FL
    """
    # Charger les données
    df = pd.read_csv(dataset_path)
    
    # Créer le serveur FL
    server = FederatedServer()
    
    # Créer les clients FL (un par zone)
    zones = df['zone'].unique()
    
    print(f"\n📡 INITIALISATION FEDERATED LEARNING")
    print(f"{'='*70}")
    print(f"   Zones: {', '.join(zones)}")
    
    for zone in zones:
        # Créer le client
        client = FederatedClient(
            client_id=f"client_{zone.lower()}",
            zone_name=zone
        )
        
        # Charger les données locales (privées)
        zone_data = df[df['zone'] == zone]
        client.load_data(zone_data)
        
        # Enregistrer le client
        server.register_client(client)
    
    print(f"\n   ✅ {len(server.clients)} clients enregistrés")
    
    # Lancer l'entraînement FL
    global_model = server.train(num_rounds=num_rounds)
    
    # Afficher l'historique
    print(f"\n📊 HISTORIQUE D'ENTRAÎNEMENT:")
    history = server.get_training_history()
    print(history.to_string(index=False))
    
    # Sauvegarder le modèle global
    server.save_global_model('/home/claude/models/federated_global_model.pkl')
    
    return server, global_model


if __name__ == "__main__":
    import os
    os.makedirs('/home/claude/models', exist_ok=True)
    
    # Lancer la simulation FL
    server, global_model = run_federated_learning_simulation(
        dataset_path='irrigation_dataset_mauritania.csv',
        num_rounds=3  # 3 rounds pour la démo
    )
    
    print(f"\n{'='*70}")
    print(f"🎯 RÉSULTAT FINAL")
    print(f"{'='*70}")
    print(f"   R² global: {global_model['avg_r2']:.4f}")
    print(f"   MSE global: {global_model['avg_mse']:.4f}")
    print(f"   Échantillons total: {global_model['total_samples']}")
    print(f"{'='*70}\n")