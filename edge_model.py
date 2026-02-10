"""
Edge Computing - Modèle de Prédiction Locale
Algorithme de régression pour prédire le temps d'arrosage nécessaire
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from datetime import datetime

class IrrigationEdgeModel:
    """
    Modèle Edge pour la prédiction du temps d'irrigation
    Déployé localement sur les capteurs IoT / Edge Nodes
    """
    
    def __init__(self, zone_name):
        self.zone_name = zone_name
        self.model = None
        self.feature_names = ['humidity', 'temperature', 'ph', 'evapotranspiration']
        self.metrics = {}
        
    def train(self, data):
        """
        Entraînement local du modèle sur les données de la zone
        
        Args:
            data: DataFrame avec les colonnes: humidity, temperature, ph, 
                  evapotranspiration, recommended_irrigation_time_min
        """
        print(f"\n🌱 Entraînement du modèle Edge pour la zone: {self.zone_name}")
        print(f"   Nombre d'échantillons: {len(data)}")
        
        # Préparation des données
        X = data[self.feature_names]
        y = data['recommended_irrigation_time_min']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraînement Random Forest (efficace pour Edge)
        self.model = RandomForestRegressor(
            n_estimators=50,  # Réduit pour performance Edge
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'zone': self.zone_name,
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'n_samples': len(data),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✅ R² Score: {self.metrics['r2']:.4f}")
        print(f"   ✅ RMSE: {self.metrics['rmse']:.4f} minutes")
        print(f"   ✅ MAE: {self.metrics['mae']:.4f} minutes")
        
        return self.metrics
    
    def predict(self, sensor_data):
        """
        Prédiction du temps d'irrigation
        
        Args:
            sensor_data: dict avec {humidity, temperature, ph, evapotranspiration}
        
        Returns:
            dict avec prédiction et recommandation
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appelez d'abord train()")
        
        # Créer DataFrame pour prédiction
        df = pd.DataFrame([sensor_data])
        df = df[self.feature_names]  # Assurer l'ordre des features
        
        # Prédiction
        irrigation_time = self.model.predict(df)[0]
        
        # Déterminer le niveau d'urgence
        if irrigation_time > 15:
            urgence = "CRITIQUE"
            action = "Irrigation immédiate requise"
        elif irrigation_time > 10:
            urgence = "ÉLEVÉE"
            action = "Planifier irrigation dans les 2h"
        elif irrigation_time > 7:
            urgence = "MOYENNE"
            action = "Irrigation dans la journée"
        else:
            urgence = "FAIBLE"
            action = "Conditions optimales"
        
        return {
            'zone': self.zone_name,
            'irrigation_time_min': round(float(irrigation_time), 2),
            'urgence': urgence,
            'action': action,
            'sensor_data': sensor_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath):
        """Sauvegarder le modèle pour déploiement Edge"""
        model_data = {
            'model': self.model,
            'zone_name': self.zone_name,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath):
        """Charger un modèle pré-entraîné"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.zone_name = model_data['zone_name']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        print(f"📥 Modèle chargé: {filepath}")
    
    def get_feature_importance(self):
        """Obtenir l'importance des features"""
        if self.model is None:
            return None
        
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def train_edge_models_per_zone(dataset_path):
    """
    Entraîner un modèle Edge pour chaque zone (Rosso, Kaedi, Boghé)
    
    Returns:
        dict: {zone_name: IrrigationEdgeModel}
    """
    import os
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Charger les données
    df = pd.read_csv(dataset_path)
    print(f"\n📊 Dataset chargé: {len(df)} enregistrements")
    print(f"   Zones: {df['zone'].unique()}")
    
    models = {}
    
    # Entraîner un modèle par zone
    for zone in df['zone'].unique():
        zone_data = df[df['zone'] == zone]
        
        model = IrrigationEdgeModel(zone_name=zone)
        metrics = model.train(zone_data)
        
        # Sauvegarder le modèle
        model_path = f'models/edge_model_{zone.lower()}.pkl'
        model.save_model(model_path)
        
        models[zone] = model
        
        # Afficher importance des features
        print(f"\n   📊 Importance des features pour {zone}:")
        for feature, importance in model.get_feature_importance().items():
            print(f"      {feature}: {importance:.4f}")
    
    return models


if __name__ == "__main__":
    # Entraîner les modèles
    models = train_edge_models_per_zone('irrigation_dataset_mauritania.csv')
    
    # Test de prédiction
    print("\n\n🧪 TEST DE PRÉDICTION")
    print("="*60)
    
    test_data = {
        'humidity': 55,
        'temperature': 35,
        'ph': 6.5,
        'evapotranspiration': 6.0
    }
    
    for zone, model in models.items():
        result = model.predict(test_data)
        print(f"\n🌍 Zone: {zone}")
        print(f"   Temps d'irrigation prédit: {result['irrigation_time_min']} min")
        print(f"   Niveau d'urgence: {result['urgence']}")
        print(f"   Action recommandée: {result['action']}")