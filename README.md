# 🛡️ Churn Sentinel : Intelligence Artificielle & Stratégie de Rétention

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## 📌 Aperçu du Projet
Ce projet déploie un système d'aide à la décision pour prédire le désabonnement des clients (Churn) dans le secteur des télécommunications. L'objectif est de passer d'une simple analyse prédictive à une **analyse prescriptive**, en suggérant des actions concrètes pour chaque profil client identifié comme "à risque".

## 🚀 Fonctionnalités Clés
- **Tableau de Bord Interactif :** Interface développée avec Streamlit pour une utilisation métier intuitive.
- **Analyse Prescriptive :** Génération de recommandations commerciales basées sur les données.
- **Interprétabilité (XAI) :** Utilisation des valeurs SHAP pour expliquer les décisions du modèle.
- **Performance Robuste :** Modèle optimisé pour maximiser le **Recall (0.79)**, assurant la détection de la majorité des clients sur le départ.

## 🧠 Architecture Technique
- **Preprocessing :** Pipeline de nettoyage, encodage catégoriel et normalisation .
- **Modélisation :** Benchmark comparatif entre Régression Logistique (équilibrée), Random Forest, XGBoost et Réseaux de Neurones .
- **Validation :** Stratified K-Fold Cross-Validation pour garantir la stabilité des résultats.

## 📊 Performance du Modèle Final
| Métrique | Score |
| :--- | :--- |
| **Recall (Classe 1)** | **0.79** |
| **F1-Score** | 0.61 |
| **Accuracy** | 0.73 |

## 🛠️ Installation
1. `git clone https://github.com/SALMA-elbardi/churn-sentinel.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

---
 *Projet réalisé par Salma el bardi - Étudiante en Ingénierie de Science des Données & IA*