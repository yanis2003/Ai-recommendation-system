import streamlit as st
import pandas as pd
from datetime import date

from svd_security_recommender import SVDSecurityRecommender


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)


@st.cache_resource
def build_model(df: pd.DataFrame):
    model = SVDSecurityRecommender(n_components=10)
    model.fit_from_long_df(df)
    return model


def main():
    st.set_page_config(page_title="AI Security Recommendation System", page_icon="🛡️")

    st.title("🛡️ AI Security Recommendation System")
    st.markdown(
        """
        Ce projet applique les **systèmes de recommandation basés sur la SVD**  
        à un cas de **sécurité réseau** : recommander les meilleures actions de sécurité  
        pour une machine donnée, à une date donnée.
        """
    )

    st.sidebar.header("⚙️ Paramètres")

    # Chargement des données
    data_path = "data/security_matrix.csv"
    df = load_data(data_path)

    # Construction du modèle
    model = build_model(df)

    st.sidebar.subheader("📅 Date de l'incident")
    incident_date = st.sidebar.date_input(
        "Sélectionnez la date",
        value=date.today()
    )

    machines = sorted(df["machine_id"].unique())
    machine_id = st.sidebar.selectbox("Machine", machines)

    top_n = st.sidebar.slider("Nombre d'actions recommandées", 1, 10, 5)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Lancer la recommandation ✅")

    st.markdown(f"**Date sélectionnée :** {incident_date}")
    st.markdown(f"**Machine sélectionnée :** `{machine_id}`")

    st.markdown("---")

    if run_button:
        st.subheader("🔐 Actions de sécurité recommandées")

        recs = model.recommend_top_n(machine_id, n=top_n)

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
            "Ces recommandations sont basées sur un modèle de **décomposition en valeurs singulières (SVD)**, "
            "similaire à celui présenté dans ton **poster universitaire sur les systèmes de recommandation**."
        )

    st.markdown("---")
    st.caption("Projet AI-Recommendation-System · SVD · Sécurité réseau & IA")


if __name__ == "__main__":
    main()
