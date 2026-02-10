"""
Fog Computing - Agrégation Régionale
Niveau intermédiaire entre Edge et Cloud
- Agrège les données des Edge Nodes
- Détecte les alertes régionales
- Réduit la latence
- Filtre les données avant envoi au Cloud
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, avg, count, max as spark_max, min as spark_min
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, MapType
import json

class FogProcessor:
    """
    Processeur Fog Computing pour agrégation régionale
    """
    
    def __init__(self, kafka_broker="localhost:9092"):
        # Initialisation Spark
        self.spark = SparkSession.builder \
            .appName("FogComputing-Mauritanie") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.kafka_broker = kafka_broker
        print(f"🌫️  Fog Computing Node initialisé")
        print(f"   Kafka Broker: {kafka_broker}")
    
    def define_schema(self):
        """Définir le schéma des données Edge"""
        return StructType([
            StructField("zone", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("node_id", StringType(), True),
            StructField("sensor_data", StructType([
                StructField("humidity", DoubleType(), True),
                StructField("temperature", DoubleType(), True),
                StructField("ph", DoubleType(), True),
                StructField("evapotranspiration", DoubleType(), True),
                StructField("irrigation_time_actual", DoubleType(), True)
            ]), True),
            StructField("edge_prediction", StructType([
                StructField("irrigation_time_predicted", DoubleType(), True),
                StructField("urgence", StringType(), True),
                StructField("edge_prediction", BooleanType(), True),
                StructField("model_r2", DoubleType(), True)
            ]), True)
        ])
    
    def read_edge_stream(self):
        """Lire le flux Kafka des Edge Nodes"""
        schema = self.define_schema()
        
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_broker) \
            .option("subscribe", "irrigation-edge-data") \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        # Parser le JSON
        parsed_df = df.selectExpr("CAST(value AS STRING) as json_value") \
            .select(from_json(col("json_value"), schema).alias("data")) \
            .select("data.*")
        
        return parsed_df
    
    def aggregate_regional_data(self, stream_df):
        """
        Agrégation régionale des données
        Calcul de statistiques par zone sur fenêtre temporelle
        """
        aggregated = stream_df \
            .groupBy(
                col("zone"),
                window(col("timestamp"), "1 minute")
            ) \
            .agg(
                count("*").alias("num_readings"),
                avg("sensor_data.humidity").alias("avg_humidity"),
                avg("sensor_data.temperature").alias("avg_temperature"),
                avg("sensor_data.ph").alias("avg_ph"),
                avg("edge_prediction.irrigation_time_predicted").alias("avg_irrigation_time"),
                spark_max("edge_prediction.irrigation_time_predicted").alias("max_irrigation_time"),
                spark_min("edge_prediction.irrigation_time_predicted").alias("min_irrigation_time")
            ) \
            .select(
                col("zone"),
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("num_readings"),
                col("avg_humidity"),
                col("avg_temperature"),
                col("avg_ph"),
                col("avg_irrigation_time"),
                col("max_irrigation_time"),
                col("min_irrigation_time")
            )
        
        return aggregated
    
    def detect_regional_alerts(self, stream_df):
        """
        Détection d'alertes régionales
        Filtre uniquement les situations critiques
        """
        alerts = stream_df \
            .filter(
                (col("edge_prediction.urgence") == "CRITIQUE") |
                (col("edge_prediction.urgence") == "ÉLEVÉE")
            ) \
            .select(
                col("zone"),
                col("timestamp"),
                col("sensor_data.humidity").alias("humidity"),
                col("sensor_data.temperature").alias("temperature"),
                col("edge_prediction.irrigation_time_predicted").alias("irrigation_needed"),
                col("edge_prediction.urgence").alias("urgence")
            )
        
        return alerts
    
    def start_processing(self):
        """Démarrer le traitement Fog"""
        print("\n🌫️  Démarrage du traitement Fog Computing...")
        print("="*70)
        
        # Lire le flux Edge
        edge_stream = self.read_edge_stream()
        
        # Stream 1: Agrégation régionale
        print("\n📊 Stream 1: Agrégation régionale (fenêtre 1 min)")
        query_aggregation = self.aggregate_regional_data(edge_stream) \
            .writeStream \
            .outputMode("complete") \
            .format("console") \
            .option("truncate", "false") \
            .queryName("fog_regional_aggregation") \
            .start()
        
        # Stream 2: Détection d'alertes
        print("🚨 Stream 2: Détection d'alertes critiques")
        query_alerts = self.detect_regional_alerts(edge_stream) \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", "false") \
            .queryName("fog_alert_detection") \
            .start()
        
        # Stream 3: Affichage détaillé de toutes les données Edge
        print("📡 Stream 3: Flux complet des données Edge")
        query_raw = edge_stream \
            .select(
                col("zone"),
                col("timestamp"),
                col("sensor_data.humidity").alias("humidity"),
                col("sensor_data.temperature").alias("temperature"),
                col("edge_prediction.irrigation_time_predicted").alias("pred_irrigation"),
                col("edge_prediction.urgence").alias("urgence")
            ) \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", "false") \
            .queryName("fog_raw_stream") \
            .start()
        
        print("\n✅ Fog Computing actif - En attente des données Edge...")
        print("   Appuyez sur Ctrl+C pour arrêter\n")
        
        # Attendre les requêtes
        try:
            query_raw.awaitTermination()
        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt du Fog Computing...")
            query_aggregation.stop()
            query_alerts.stop()
            query_raw.stop()
            self.spark.stop()


if __name__ == "__main__":
    fog = FogProcessor()
    fog.start_processing()