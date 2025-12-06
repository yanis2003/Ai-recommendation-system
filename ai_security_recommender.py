import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. SYSTEME DE RECOMMANDATION DE SECURITE
# -------------------------------

class AISecurityRecommender:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.actions_map = {
            0: "Aucune action nécessaire",
            1: "Bloquer l'adresse IP source",
            2: "Isoler la machine compromise",
            3: "Fermer le port suspect",
            4: "Augmenter la surveillance du trafic réseau",
            5: "Scanner la machine pour vérifier une infection"
        }

    def train(self, df):
        X = df.drop(["attack_type"], axis=1)
        y = df["attack_type"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        print("Accuracy :", self.model.score(X_test, y_test))

    def recommend_action(self, features):
        prediction = self.model.predict([features])[0]
        return self.actions_map[prediction]


# -------------------------------
# 2. DEMO AVEC UN PETIT DATASET
# -------------------------------

if __name__ == "__main__":
    # Dataset d’exemple (normalement tu importes un vrai dataset)
    df = pd.DataFrame({
        "duration": [10, 5, 100, 1, 60],
        "src_bytes": [200, 50, 5000, 10, 1200],
        "dst_bytes": [300, 20, 1, 5, 80],
        "failed_logins": [0, 0, 3, 0, 1],
        "attack_type": [0, 0, 2, 0, 1]
    })

    recommender = AISecurityRecommender()
    recommender.train(df)

    # Exemple : un événement réseau à analyser
    sample = [80, 200, 30, 1]  # features
    action = recommender.recommend_action(sample)

    print("\n🚨 Action recommandée :", action)
