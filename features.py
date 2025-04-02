import pandas as pd
import streamlit as st
import time
import random
import json
from datetime import datetime, timedelta
import numpy as np
from feat_egg import calculate_winrates

features_used = [
    'Home Team', 'Away Team', 'Match Excitement', 'Home Team Rating',
    'Away Team Rating', 'Home Team Possession %', 'Away Team Possession %',
    'Home Team Off Target Shots', 'Home Team On Target Shots',
    'Home Team Total Shots', 'Home Team Blocked Shots', 'Home Team Corners',
    'Home Team Throw Ins', 'Home Team Pass Success %',
    'Home Team Aerials Won', 'Home Team Clearances', 'Home Team Fouls',
    'Home Team Yellow Cards', 'Home Team Second Yellow Cards',
    'Home Team Red Cards', 'Away Team Off Target Shots',
    'Away Team On Target Shots', 'Away Team Total Shots',
    'Away Team Blocked Shots', 'Away Team Corners', 'Away Team Throw Ins',
    'Away Team Pass Success %', 'Away Team Aerials Won',
    'Away Team Clearances', 'Away Team Fouls', 'Away Team Yellow Cards',
    'Away Team Second Yellow Cards', 'Away Team Red Cards', 'year',
    'Home Team Half Time Goals', 'Away Team Half Time Goals',
    'away_team_away_winrate', 'away_team_overall_winrate',
    'home_team_home_winrate', 'home_team_overall_winrate'
]


@st.cache_data
def get_info():
    data = pd.read_csv('data/combined_data_laliga.csv')
    teams = set(data['Home Team'].unique())
    t2i = {team: i for i, team in enumerate(teams)}
    i2t = {i: team for team, i in t2i.items()}
    return data, t2i, i2t

data, t2i, i2t = get_info()


from features import *

class FastForwardMatchSimulation:
    def __init__(self, general_inputs = {}):
        self.match_minute = 0
        self.is_match_active = False
        self.home_score = 0
        self.away_score = 0
        self.last_stats = None
        self.general_inputs = general_inputs

        if 'home_team_rating' not in st.session_state:
            st.session_state.home_team_rating = 7.0
        if 'away_team_rating' not in st.session_state:
            st.session_state.away_team_rating = 7.0
        
    def generate_match_stats(self):
        """Generate progressive match statistics"""
        if self.last_stats is None:
            home_possession = random.uniform(45, 55)
            self.last_stats = {
                'home_possession': home_possession,
                'home_shots': 0,
                'away_shots': 0,
                'home_corners': 0,
                'away_corners': 0,
                'home_fouls': 0,
                'away_fouls': 0,
                'home_on_target': 0,
                'away_on_target': 0
            }
        
        home_rating = st.session_state.get('home_team_rating', 7.0)
        away_rating = st.session_state.get('away_team_rating', 7.0)
        if random.random() < 0.5:  
            event_type = random.choice(['shot', 'corner', 'foul', 'goal'])
            team = random.choice(['home', 'away'])
            
            print(home_rating, away_rating)
            
            if event_type == 'shot':
                self.last_stats[f'{team}_shots'] += 1
                
                
                base_accuracy = 0.4 
                if team == 'home':
                    shot_accuracy = base_accuracy * (home_rating / away_rating)
                else:
                    shot_accuracy = base_accuracy * (away_rating / home_rating)
                
                if random.random() < shot_accuracy:  
                    self.last_stats[f'{team}_on_target'] += 1
                    
                    base_goal_prob = 0.2  
                    if team == 'home':
                        goal_probability = base_goal_prob * (home_rating / away_rating)
                    else:
                        goal_probability = base_goal_prob * (away_rating / home_rating)
                    
                    if random.random() < goal_probability:
                        if team == 'home':
                            self.home_score += 1
                        else:
                            self.away_score += 1
            elif event_type == 'corner':
                self.last_stats[f'{team}_corners'] += 1
            elif event_type == 'foul':
                self.last_stats[f'{team}_fouls'] += 1
        
        home_rating = st.session_state.get('home_team_rating', 7.0)
        away_rating = st.session_state.get('away_team_rating', 7.0)
        rating_factor = (home_rating / away_rating) - 1  
        base_change = random.uniform(-2, 2)
        possession_change = base_change + (rating_factor * 0.5) 

        
        self.last_stats['home_possession'] = max(min(
            self.last_stats['home_possession'] + possession_change, 
            75 
        ), 25)
        
        return {
            'match_minute': self.match_minute,
            'home_score': self.home_score,
            'away_score': self.away_score,
            'home_team': {
                'possession': self.last_stats['home_possession'],
                'shots_total': self.last_stats['home_shots'],
                'shots_on_target': self.last_stats['home_on_target'],
                'corners': self.last_stats['home_corners'],
                'fouls': self.last_stats['home_fouls']
            },
            'away_team': {
                'possession': 100 - self.last_stats['home_possession'],
                'shots_total': self.last_stats['away_shots'],
                'shots_on_target': self.last_stats['away_on_target'],
                'corners': self.last_stats['away_corners'],
                'fouls': self.last_stats['away_fouls']
            }
        }

def adjust_predictions_with_score(probabilities, home_score, away_score, minute):
    """Adjust win probabilities based on current score and time"""
    goal_diff = home_score - away_score
    time_weight = minute/90 
    score_weight = 0.3  
    
    if goal_diff > 0:
        home_advantage = goal_diff * time_weight * score_weight
        adjusted_probs = [
            probabilities[0] * (1 - home_advantage), 
            probabilities[1] * (1 + home_advantage), 
            probabilities[2] * (1 - home_advantage) 
        ]
    elif goal_diff < 0:
        away_advantage = abs(goal_diff) * time_weight * score_weight
        adjusted_probs = [
            probabilities[0] * (1 - away_advantage),
            probabilities[1] * (1 - away_advantage),
            probabilities[2] * (1 + away_advantage)
        ]
    else:
        adjusted_probs = probabilities

    total = sum(adjusted_probs)
    adjusted_probs = [p/total for p in adjusted_probs]
    
    return adjusted_probs

def simulate_match(simulation, placeholder, model, speed=0.25):
    """Run match simulation with progress bar and live predictions"""
    progress_bar = st.progress(0)
    prediction_placeholder = st.empty()
    
    for minute in range(91):
        simulation.match_minute = minute
        stats = simulation.generate_match_stats()
        
        # Update display
        placeholder.empty()
        with placeholder.container():
            col1, col2, col3 = st.columns([2,1,2])
            
            with col1:
                st.metric("Home Team", f"Shots: {stats['home_team']['shots_total']}")
                st.metric("Possession", f"{stats['home_team']['possession']:.1f}%")
                st.metric("Corners", stats['home_team']['corners'])
            
            with col2:
                st.metric("⏱️ Minute", minute)
                st.metric("Score", f"{stats['home_score']} - {stats['away_score']}")
            
            with col3:
                st.metric("Away Team", f"Shots: {stats['away_team']['shots_total']}")
                st.metric("Possession", f"{stats['away_team']['possession']:.1f}%")
                st.metric("Corners", stats['away_team']['corners'])
        
        current_features = process_final_stats(stats)
        features_df = pd.DataFrame([current_features])[model.feature_names_in_]
        
        prediction_placeholder.empty()
        with prediction_placeholder.container():
            st.subheader("Live Prediction")
            probabilities = model.predict_proba(features_df)[0]
            probabilities = adjust_predictions_with_score(
                                probabilities, 
                                stats['home_score'],
                                stats['away_score'],
                                minute
                            )
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                st.metric("Home Win", f"{probabilities[1]*100:.1f}%")
            with pred_col2:
                st.metric("Draw", f"{probabilities[0]*100:.1f}%")
            with pred_col3:
                st.metric("Away Win", f"{probabilities[2]*100:.1f}%")
        
        progress_bar.progress(minute/90)
        time.sleep(speed)
    
    return stats


def get_features(simulation, model):
    st.sidebar.title('Live Match Predictor')
    st.sidebar.subheader('Match Control')
    
    stats_placeholder = st.empty()
    
    if not simulation.is_match_active:
        if st.button('Start Match'):
            simulation.is_match_active = True
            simulation.match_minute = 0
            simulation.home_score = 0
            simulation.away_score = 0
            simulation.last_stats = None
            
            speed = st.sidebar.slider(
                'Simulation Speed (seconds)', 
                min_value=0.1, 
                max_value=1.0, 
                value=0.25,
                step=0.05,
                help="Adjust how fast the match simulation runs"
            )
            
            final_stats = simulate_match(simulation, stats_placeholder, model, speed)
            simulation.is_match_active = False
            return process_final_stats(final_stats)
    else:
        if st.button('End Match'):
            simulation.is_match_active = False
    
    general_inputs = {}
    sorted_teams = sorted(list(t2i.keys()))
    barcelona_index = sorted_teams.index('Barcelona') if 'Barcelona' in sorted_teams else 0
    
    home_team_index = st.sidebar.selectbox(
        "Select Home Team",
        options=sorted(list(t2i.keys())),
        key="home_team_select",
        index=4
    )
    
    away_team_index = st.sidebar.selectbox(
        "Select Away Team",
        options=sorted(list(t2i.keys())),
        key="away_team_select"
    )
    
    # Team ratings
    home_team_rating = st.sidebar.slider(
        "Home Team Rating",
        min_value=0.0,
        max_value=10.0,
        value=9.0,
        key="home_team_rating_slider"
    )
    
    away_team_rating = st.sidebar.slider(
        "Away Team Rating",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        key="away_team_rating_slider"
    )
    
    st.session_state['home_team_index'] = t2i[home_team_index]
    st.session_state['away_team_index'] = t2i[away_team_index]
    st.session_state['home_team_rating'] = home_team_rating
    st.session_state['away_team_rating'] = away_team_rating
    
    
    general_inputs['Home Team'] = t2i[home_team_index]
    general_inputs['Away Team'] = t2i[away_team_index]
    
    return get_manual_features(general_inputs)


    

def process_final_stats(stats):
    """Convert final match stats into feature dictionary"""
    features = {}

    home_team = i2t[st.session_state.get('home_team_index', 0)]
    away_team = i2t[st.session_state.get('away_team_index', 1)]
    
    home_home_winrate, home_overall_winrate = calculate_winrates(data, home_team, is_home=True)
    away_away_winrate, away_overall_winrate = calculate_winrates(data, away_team, is_home=False)
    
    features.update({
        'home_team_home_winrate': home_home_winrate,
        'home_team_overall_winrate': home_overall_winrate,
        'away_team_away_winrate': away_away_winrate,
        'away_team_overall_winrate': away_overall_winrate
    })
    
    features['Home Team'] = st.session_state.get('home_team_index', 0)
    features['Away Team'] = st.session_state.get('away_team_index', 1)
    features['Home Team Rating'] = st.session_state.get('home_team_rating', 7.0)
    features['Away Team Rating'] = st.session_state.get('away_team_rating', 7.0)
    
    features.update({
        'year': 2024,
        'Match Excitement': random.uniform(6.0, 9.0),
        'Home Team Possession %': stats['home_team']['possession'],
        'Away Team Possession %': stats['away_team']['possession'],
        'Home Team Total Shots': stats['home_team']['shots_total'],
        'Away Team Total Shots': stats['away_team']['shots_total'],
        'Home Team On Target Shots': stats['home_team']['shots_on_target'],
        'Away Team On Target Shots': stats['away_team']['shots_on_target'],
        'Home Team Off Target Shots': stats['home_team']['shots_total'] - stats['home_team']['shots_on_target'],
        'Away Team Off Target Shots': stats['away_team']['shots_total'] - stats['away_team']['shots_on_target'],
        'Home Team Corners': stats['home_team']['corners'],
        'Away Team Corners': stats['away_team']['corners'],
        'Home Team Fouls': stats['home_team']['fouls'],
        'Away Team Fouls': stats['away_team']['fouls'],
        'Home Team Blocked Shots': int(random.uniform(1, 5)),
        'Away Team Blocked Shots': int(random.uniform(1, 5)),
        'Home Team Throw Ins': int(random.uniform(10, 25)),
        'Away Team Throw Ins': int(random.uniform(10, 25)),
        'Home Team Pass Success %': random.uniform(75, 90),
        'Away Team Pass Success %': random.uniform(75, 90),
        'Home Team Aerials Won': int(random.uniform(10, 25)),
        'Away Team Aerials Won': int(random.uniform(10, 25)),
        'Home Team Clearances': int(random.uniform(10, 25)),
        'Away Team Clearances': int(random.uniform(10, 25)),
        'Home Team Yellow Cards': int(random.uniform(0, 3)),
        'Away Team Yellow Cards': int(random.uniform(0, 3)),
        'Home Team Second Yellow Cards': 0,
        'Away Team Second Yellow Cards': 0,
        'Home Team Red Cards': 0,
        'Away Team Red Cards': 0,
        'Home Team Half Time Goals': stats['home_score'],
        'Away Team Half Time Goals': stats['away_score']
    })
    
    return features

def get_manual_features(general_inputs):
    """Generate features without UI elements"""
    team_inputs = {}

    home_team = i2t[st.session_state.get('home_team_index', 0)]
    away_team = i2t[st.session_state.get('away_team_index', 1)]
    
    # Calculate and add winrates
    home_home_winrate, home_overall_winrate = calculate_winrates(data, home_team, is_home=True)
    away_away_winrate, away_overall_winrate = calculate_winrates(data, away_team, is_home=False)
    team_inputs['Home Team Rating'] = st.session_state.get('home_team_rating', 7.0)
    team_inputs['Away Team Rating'] = st.session_state.get('away_team_rating', 7.0)
    
    # Set default values for general features
    general_features = ['Match Excitement', 'year']
    for feature in general_features:
        if feature == 'year':
            general_inputs[feature] = 2024
        else:
            general_inputs[feature] = 5.0  # Default value without slider
    
    # Set default values for team features
    team_features = [f for f in features_used if f not in general_features and f not in ['Home Team Rating', 'Away Team Rating', 'Home Team', 'Away Team']]
    for feature in team_features:
        if 'Possession' in feature or 'Pass Success' in feature:
            team_inputs[feature] = 50.0  # Default percentage
        elif any(card in feature for card in ['Yellow Cards', 'Red Cards', 'Second Yellow Cards']):
            team_inputs[feature] = 0  # Default card count
        elif any(stat in feature for stat in ['Shots', 'Corners', 'Throw Ins', 'Aerials Won', 'Clearances', 'Fouls']):
            team_inputs[feature] = 5  # Default stat count
        elif 'Goals' in feature:
            team_inputs[feature] = 0  # Default goals

    # Combine all features
    features = {**general_inputs, **team_inputs}
    features.update({
        'home_team_home_winrate': home_home_winrate,
        'home_team_overall_winrate': home_overall_winrate,
        'away_team_away_winrate': away_away_winrate,
        'away_team_overall_winrate': away_overall_winrate
    })
    
    return features

def get_manual_features(general_inputs):
    """Fallback to manual feature input when match is not active"""
    team_inputs = {}

    home_team = i2t[st.session_state.get('home_team_index', 0)]
    away_team = i2t[st.session_state.get('away_team_index', 1)]
    
    # Calculate and add winrates
    home_home_winrate, home_overall_winrate = calculate_winrates(data, home_team, is_home=True)
    away_away_winrate, away_overall_winrate = calculate_winrates(data, away_team, is_home=False)
    
    team_inputs.update({
        'home_team_home_winrate': home_home_winrate,
        'home_team_overall_winrate': home_overall_winrate,
        'away_team_away_winrate': away_away_winrate,
        'away_team_overall_winrate': away_overall_winrate
    })
    
    team_inputs['Home Team Rating'] = st.session_state.get('home_team_rating', 7.0)
    team_inputs['Away Team Rating'] = st.session_state.get('away_team_rating', 7.0)
    team_inputs['Home Team'] = st.session_state.get('home_team_index', 0)
    team_inputs['Away Team'] = st.session_state.get('away_team_index', 1)
    
    general_features = ['Match Excitement', 'year']
    for feature in general_features:
        if feature == 'year':
            general_inputs[feature] = 2024
        else:
            general_inputs[feature] = st.sidebar.slider(
                feature, min_value=0.0, max_value=10.0, value=5.0)
    
    team_features = [f for f in features_used if f not in general_features and f not in ['Home Team Rating', 'Away Team Rating', 'Home Team', 'Away Team']]
    for feature in team_features:
        if 'Possession' in feature or 'Pass Success' in feature:
            team_inputs[feature] = st.sidebar.slider(
                feature, min_value=0.0, max_value=100.0, value=50.0)
        elif any(card in feature for card in ['Yellow Cards', 'Red Cards', 'Second Yellow Cards']):
            team_inputs[feature] = st.sidebar.number_input(
                feature, min_value=0, max_value=5, value=0)
        elif any(stat in feature for stat in ['Shots', 'Corners', 'Throw Ins', 'Aerials Won', 'Clearances', 'Fouls']):
            team_inputs[feature] = st.sidebar.number_input(
                feature, min_value=0, max_value=30, value=5)
        elif 'Goals' in feature:
            team_inputs[feature] = st.sidebar.number_input(
                feature, min_value=0, max_value=5, value=0)
            
    return {**general_inputs, **team_inputs}