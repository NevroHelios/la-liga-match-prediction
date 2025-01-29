import pandas as pd

def calculate_winrates(data, team, is_home=True):
    """Calculate home/away and overall winrates for a team"""
    data['Home Team Goals'] = data['Score'].apply(lambda x: x.split('-')[0]).astype(int)
    data['Away Team Goals'] = data['Score'].apply(lambda x: x.split('-')[1]).astype(int)
    if is_home:
        home_matches = data[data['Home Team'] == team].tail(10)
        
        home_wins = len(home_matches[home_matches['Home Team Goals'] > home_matches['Away Team Goals']])
        home_winrate = home_wins / len(home_matches) if len(home_matches) > 0 else 0.5
        
        all_matches = pd.concat([
            data[data['Home Team'] == team],
            data[data['Away Team'] == team]
        ]).tail(10)
        total_wins = len(all_matches[
            ((all_matches['Home Team'] == team) & (all_matches['Home Team Goals'] > all_matches['Away Team Goals'])) |
            ((all_matches['Away Team'] == team) & (all_matches['Away Team Goals'] > all_matches['Home Team Goals']))
        ])
        overall_winrate = total_wins / len(all_matches) if len(all_matches) > 0 else 0.5
        
        return home_winrate, overall_winrate
    else:
        away_matches = data[data['Away Team'] == team].tail(10)
        away_wins = len(away_matches[away_matches['Away Team Goals'] > away_matches['Home Team Goals']])
        away_winrate = away_wins / len(away_matches) if len(away_matches) > 0 else 0.5
        
        all_matches = pd.concat([
            data[data['Home Team'] == team],
            data[data['Away Team'] == team]
        ]).tail(10)
        total_wins = len(all_matches[
            ((all_matches['Home Team'] == team) & (all_matches['Home Team Goals'] > all_matches['Away Team Goals'])) |
            ((all_matches['Away Team'] == team) & (all_matches['Away Team Goals'] > all_matches['Home Team Goals']))
        ])
        overall_winrate = total_wins / len(all_matches) if len(all_matches) > 0 else 0.5
        
        return away_winrate, overall_winrate