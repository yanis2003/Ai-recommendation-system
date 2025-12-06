# AI-Recommendation-System 🛡️

Ce projet est un **système de recommandation basé sur la SVD**, appliqué à la **sécurité réseau**.

Il fait suite à un **poster universitaire** que j’ai réalisé sur les **systèmes de recommandation** et la **décomposition en valeurs singulières (SVD)**.  
L’objectif est de passer du concept théorique à un **projet concret** que les entreprises peuvent comprendre et utiliser.

##  Idée générale

- Les **lignes** représentent des machines / hôtes (`machine_id`)
- Les **colonnes** représentent des actions de sécurité (`action_id`)
- La **valeur** est un score d’efficacité ou de pertinence
- La SVD permet d’extraire des **facteurs latents** et de prédire
  quelles actions sont les plus adaptées à une machine donnée.

## Modèle

Le moteur est implémenté dans `svd_security_recommender.py` :

- SVD tronquée (scikit-learn `TruncatedSVD`)
- Matrice complétée avec la moyenne globale
- Recommandation des actions avec score prédictif

##  Interface utilisateur

L’interface graphique est réalisée avec **Streamlit** (`app.py`) :

- Choix de la **date d’incident**
- Sélection de la **machine**
- Nombre d’actions recommandées
- Affichage d’un tableau avec les **meilleures actions de sécurité**.

## ▶Lancer le projet

```bash
pip install -r requirements.txt
streamlit run app.py
