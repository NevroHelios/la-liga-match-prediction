import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from model import FootballPredictorWrapper

score_dict = {
    0: "Draw",
    1: "Home Team Wins",
    2: "Away Team Wins",
}

st.set_page_config(
    page_title="La Liga Match Predictor",
    page_icon="⚽",
    layout="wide"
)

from features import *

st.markdown("""
    <style>
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .prediction {
        font-size: 24px;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        color: #fff;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">⚽ La Liga Match Predictor</p>', unsafe_allow_html=True)

tab_details, tab_playground = st.tabs(["Project Details", "Model Playground"])

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #f39c12;
    }
    .subtitle {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    #box {
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        color: #ffffff;
    }
    .highlight {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

with tab_details:
    st.image("https://wallpapers.com/images/hd/bocchi-the-rock-surprised-reaction-uf98jl11d34v5h6c.jpg", use_container_width=True)


with tab_playground:
    tab_prediction, tab_stats = st.tabs(["Prediction", "Match Statistics"])

    @st.cache_resource
    def load_model(torch_model=False, device='cpu'):
        if torch_model:
            # model = torch.load('saved_models/football_predictor.pth')
            return FootballPredictorWrapper(device='cpu')
        return pickle.load(open('saved_models/model_xgb.pkl', 'rb')).best_estimator_

    with tab_prediction:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Enter Match Details")
            with st.spinner('Loading data...'):
                data, t2i, i2t = get_info()
            model_type = st.selectbox(
                "Select Model Type",
                ["Extreme Gradient Boosting", "Neural Network"],
                key="model_type"
            )

            clf = load_model(torch_model=(model_type == "Neural Network"), device='cpu')
            try:
                clf.set_params(device='cpu')
            except:
                ...
            if 'simulation' not in st.session_state:
                st.session_state.simulation = FastForwardMatchSimulation()
            features_dict = get_features(st.session_state.simulation, clf)  

            # Get features with model passed as parameter
            features_df = pd.DataFrame([features_dict])[clf.feature_names_in_]
            # print(features_dict)
            # print("-------------------")
            # print(clf.feature_names_in_)

            result = clf.predict(features_df)
            proba = clf.predict_proba(features_df)[0]
            class_labels = clf.classes_ 

            label_to_prob = dict(zip(class_labels, proba))
            print(label_to_prob)
            
            predicted_label = result[0]  # -1 or 0 or 1
            predicted_description = score_dict[predicted_label]
            predicted_confidence = label_to_prob[predicted_label] * 100

            result_color = {
                1: '#28a745',  # Green = home
                0: '#17a2b8',  # Yellow = draw
                2: '#dc3545'  # Red = away
            }

            st.markdown(
                f'<div class="prediction" style="background-color: {result_color[predicted_label]};">'
                f'Predicted Result: {predicted_description}<br>'
                f'Confidence: {predicted_confidence:.1f}%'
                '</div>',
                unsafe_allow_html=True
            )

        with col_right:
            st.subheader("Win Probabilities")
            prob_display = {
                "Away Win": label_to_prob.get(2, 0),
                "Draw": label_to_prob.get(0, 0),
                "Home Win": label_to_prob.get(1, 0)
            }
            fig = px.pie(
                names=prob_display.keys(),
                values=prob_display.values(),
                title="Match Outcome Probabilities",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_stats:
        st.subheader("Team Statistics Comparison")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Possession")
            possession_data = {
                'Team': ['Home', 'Away'],
                'Possession': [
                    features_dict['Home Team Possession %'],
                    features_dict['Away Team Possession %']
                ]
            }
            fig_poss = px.bar(possession_data, x='Team', y='Possession', 
                            title='Possession %', color='Team', text='Possession')
            fig_poss.update_traces(textposition='outside')
            st.plotly_chart(fig_poss, use_container_width=True)
            
        with col_right:
            st.subheader("Shots")
            shots_data = pd.DataFrame({
                'Team': ['Home', 'Away'],
                'On Target': [features_dict['Home Team On Target Shots'],
                            features_dict['Away Team On Target Shots']],
                'Off Target': [features_dict['Home Team Off Target Shots'],
                            features_dict['Away Team Off Target Shots']]
            })
            fig_shots = px.bar(
                shots_data, x='Team',
                y=['On Target', 'Off Target'],
                barmode='group',
                title='Shot Statistics'
            )
            st.plotly_chart(fig_shots, use_container_width=True)