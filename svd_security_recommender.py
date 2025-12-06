import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class SVDSecurityRecommender:
    """
    Système de recommandation de mesures de sécurité basé sur SVD.
    - 'machine_id' : identifiant de la machine / serveur
    - 'action_id'  : identifiant de l'action de sécurité
    - 'score'      : efficacité / pertinence de l'action (ex : 1 à 5)
    """

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=42)

        self.global_mean = None
        self.user_factors = None
        self.item_factors = None

        self.machine_ids = None
        self.action_ids = None

        # Description des actions possibles (exemple, à adapter)
        self.action_descriptions = {
            0: "Aucune action critique",
            1: "Bloquer l'adresse IP source",
            2: "Isoler la machine du réseau",
            3: "Fermer le port/protocole suspect",
            4: "Augmenter la surveillance (IDS / SIEM / logs)",
            5: "Scanner la machine (antivirus / EDR)",
        }

    def fit_from_long_df(self, df: pd.DataFrame):
        """
        df doit contenir les colonnes :
        - machine_id
        - action_id
        - score
        """
        required = {"machine_id", "action_id", "score"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Le fichier CSV doit contenir au moins les colonnes : {required}. "
                f"Colonnes trouvées : {list(df.columns)}"
            )

        # pivot -> matrice machines x actions
        pivot = df.pivot_table(
            index="machine_id",
            columns="action_id",
            values="score",
            aggfunc="mean",
            fill_value=0.0,
        )

        self.machine_ids = list(pivot.index)
        self.action_ids = list(pivot.columns)

        R = pivot.values.astype(float)
        non_zero = R[R > 0]
        if non_zero.size == 0:
            self.global_mean = 0.0
        else:
            self.global_mean = float(non_zero.mean())

        # remplace les 0 par la moyenne globale
        R_filled = np.where(R == 0, self.global_mean, R)

        # SVD tronquée
        self.user_factors = self.model.fit_transform(R_filled)
        self.item_factors = self.model.components_.T

        print("✔ Modèle SVD entraîné")
        print("  - Machines :", len(self.machine_ids))
        print("  - Actions  :", len(self.action_ids))
        print("  - Variance expliquée ~", round(self.model.explained_variance_ratio_.sum(), 3))

    def _get_machine_index(self, machine_id):
        return self.machine_ids.index(machine_id)

    def _get_action_index(self, action_id):
        return self.action_ids.index(action_id)

    def predict_score(self, machine_id, action_id) -> float:
        """
        Score prédit (plus il est élevé, plus l'action est pertinente).
        """
        if machine_id not in self.machine_ids or action_id not in self.action_ids:
            return self.global_mean

        u_idx = self._get_machine_index(machine_id)
        i_idx = self._get_action_index(action_id)

        score = self.global_mean + float(
            np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        )
        return float(score)

    def recommend_top_n(self, machine_id, n: int = 5):
        """
        Renvoie les n meilleures actions recommandées pour une machine donnée.
        Retourne une liste (action_id, score, description).
        """
        if machine_id not in self.machine_ids:
            raise ValueError(f"Machine {machine_id} inconnue du modèle.")

        results = []
        for action_id in self.action_ids:
            score = self.predict_score(machine_id, action_id)
            desc = self.action_descriptions.get(action_id, f"Action {action_id}")
            results.append((action_id, score, desc))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
