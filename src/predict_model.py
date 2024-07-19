import os
import joblib

def load_pipeline():
    """
    Charge le pipeline de prétraitement des données.

    Le chemin du fichier du pipeline est calculé par rapport à l'emplacement actuel du script.
    
    Returns:
        Le pipeline de prétraitement des données chargé à partir du fichier.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(base_path, 'models', 'preprocessor.joblib')
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    return joblib.load(pipeline_path)

def load_model():
    """
    Charge le modèle de machine learning.

    Le chemin du fichier du modèle est calculé par rapport à l'emplacement actuel du script.
    
    Returns:
        Le modèle de machine learning chargé à partir du fichier.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_v2.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_feature_names():
    """
    Charge la liste des noms des features.

    Le chemin du fichier des noms de features est calculé par rapport à l'emplacement actuel du script.
    
    Returns:
        Une liste des noms des features.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    feature_names_path = os.path.join(base_path, 'models', 'feature_names.txt')
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    with open(feature_names_path, "r") as f:
        feature_names = f.read().splitlines()
    return feature_names

def predict(model, data):
    """
    Effectue des prédictions à partir des données en utilisant le modèle fourni.

    Args:
        model: Le modèle de machine learning utilisé pour les prédictions.
        data: Les données prétraitées sur lesquelles effectuer les prédictions.
    
    Returns:
        Tuple:
            - Les prédictions du modèle.
            - Les probabilités associées à chaque classe pour chaque prédiction.
    """
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return predictions, probabilities
