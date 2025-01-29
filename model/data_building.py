import pandas 

def get_match_result(home_goals, away_goals):
    """
    0 -> draw; 1 -> home team won; 2 -> away team won
    """
    if home_goals == away_goals:
        return 0
    elif home_goals > away_goals:
        return 1
    else:
        return 2



def get_data():
    data = pandas.read_csv('../data/combined_data_laliga.csv')
    data['Home Team Half Time Goals'] = data['Half Time Score'].apply(lambda x: x.split('-')[0]).astype(int)
    data['Away Team Half Time Goals'] = data['Half Time Score'].apply(lambda x: x.split('-')[1]).astype(int)
    teams = set(data['Home Team'].unique())
    t2i = {team: i for i, team in enumerate(teams)}
    i2t = {i: team for team, i in t2i.items()}

    data['Home Team'] = data['Home Team'].map(t2i)
    data['Away Team'] = data['Away Team'].map(t2i)
    data.drop(columns=['Half Time Score', 'Score'], inplace=True, errors='ignore')
    data['result'] = data.apply(
        lambda row: get_match_result(
            row['Home Team Goals Scored'], 
            row['Away Team Goals Scored']
        ), 
        axis=1
    )
    train = data[data['year'] < 2020]
    test = data[data['year'] == 2020]
    x_train, y_train = train.drop(columns=['result', 'Unnamed: 0']), train['result']
    x_test, y_test = test.drop(columns=['result', 'Unnamed: 0']), test['result']
    return x_train, y_train, x_test, y_test, i2t, t2i