import pickle
import os
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from config import MODEL_DIR # type: ignore

def train_and_save_model(X, y, phase=2):
    # Splits exactly 15% exclusively for Validation Testing!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64, 32), activation='relu', solver='adam', learning_rate='adaptive', max_iter=2000)
    mlp.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, f"ann_model_phase_{phase}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(mlp, f)
        
    score = mlp.score(X_test, y_test)
    return mlp, score, model_path

def load_model(phase=2):
    model_path = os.path.join(MODEL_DIR, f"ann_model_phase_{phase}.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None
