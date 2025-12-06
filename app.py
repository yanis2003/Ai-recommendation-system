import streamlit as st
import pandas as pd
from datetime import date

from svd_security_recommender import SVDSecurityRecommender


st.set_page_config(
    page_title="AI Security Recommendation System",
    page_icon="🛡️",
    layout="wide",
)


def main():
    st.title("🛡️ AI Security Recommendation System")
    st.markdown(
        """
        Cette application permet à une entreprise de **déposer ses propres données**
        (format CSV) et d'obtenir des **recommandations d'actions de sécurité**
        basées sur un modèle de **SVD (système de recommandation)**.

        ✅ Suite logique d’un travail universitaire (poster sur les systèmes de recommandation & SVD).
        """
    )

    st.sidebar.header("📂 Import des données")

    uploaded_file = st.sidebar.file_uploader(
        "Déposez votre fichier CSV (colonnes : machine_id, action_id, score)",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info(
            "Veuillez déposer un fichier CSV dans la barre latérale pour commencer.\n\n"
            "Format attendu : **machine_id, action_id, score**.\n"
            "Chaque ligne représente l'efficacité d'une action de sécurité sur une machine."
        )
        return

    # Lecture des données
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return

    st.subheader("📊 Aperçu des données")
    st.dataframe(df.head())

    # Vérification des colonnes
    required_cols = {"machine_id", "action_id", "score"}
    if not required_cols.issubset(df.columns):
        st.error(
            f"Le fichier doit contenir les colonnes : {required_cols}. "
            f"Colonnes trouvées : {list(df.columns)}"
        )
        return

    # Zone de paramètres
    st.sidebar.header("⚙️ Paramètres de recommandation")

    # Date (juste pour le contexte métier, non utilisée dans le calcul pour l'instant)
    incident_date = st.sidebar.date_input(
        "Date de l'incident ou de l'analyse",
        value=date.today()
    )

    machines = sorted(df["machine_id"].unique())
    machine_id = st.sidebar.selectbox("Machine à analyser", machines)

    top_n = st.sidebar.slider("Nombre d'actions à recommander", 1, 10, 5)

    train_button = st.sidebar.button("Entraîner le modèle & recommander ✅")

    if not train_button:
        st.warning("Cliquez sur **Entraîner le modèle & recommander** pour lancer l'analyse.")
        return

    # Entraînement du modèle
    try:
        model = SVDSecurityRecommender(n_components=10)
        model.fit_from_long_df(df)
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement du modèle : {e}")
        return

    st.markdown("---")
    st.subheader("🔐 Résultats de la recommandation")

    st.write(f"**Date sélectionnée :** {incident_date}")
    st.write(f"**Machine analysée :** `{machine_id}`")

    try:
        recs = model.recommend_top_n(machine_id, n=top_n)
    except Exception as e:
        st.error(f"Erreur lors de la recommandation : {e}")
        return

    results_df = pd.DataFrame(
        [
            {
                "action_id": action_id,
                "score_prédit": round(score, 3),
                "description": desc,
            }
            for (action_id, score, desc) in recs
        ]
    )

    st.table(results_df)

    st.info(
        "Ces recommandations sont calculées à partir d’un modèle de **décomposition en valeurs singulières (SVD)**, "
        "comme présenté dans le **poster universitaire sur les systèmes de recommandation**."
    )

    st.markdown("---")
    st.caption("Projet AI-Recommendation-System · IA + Réseaux + Sécurité · Streamlit")


if __name__ == "__main__":
    main()
