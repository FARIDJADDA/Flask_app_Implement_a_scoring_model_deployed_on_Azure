from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le dossier src au chemin d'accès
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict_model import load_pipeline, load_model, load_feature_names, predict

app = Flask(__name__)

# Charger le pipeline et le modèle une seule fois au démarrage de l'application
pipeline = load_pipeline()
model = load_model()
feature_names = load_feature_names()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        # Récupérer les données de la requête
        data = request.get_json(force=True)
        input_data = np.array(data['input']).reshape(1, -1)
        
        # Convertir en DataFrame avec les noms des colonnes
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        # Prétraiter les données
        preprocessed_data = pipeline.transform(input_df)
        
        # Effectuer les prédictions
        predictions, probabilities = predict(model, preprocessed_data)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
