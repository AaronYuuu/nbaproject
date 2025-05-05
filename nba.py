import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# ----- Stats for individual team from season -----
def get_stats(df, home, away):
    """
    look through csv file for team1 and team2
    """
    dmap = {'Cleveland Cavaliers': 'CLE', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'New Orleans Pelicans': 'NOP', 
            'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 
            'Portland Trail Blazers': 'POR', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Denver Nuggets': 'DEN', 
            'Golden State Warriors': 'GSW', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Minnesota Timberwolves': 'MIN', 'Brooklyn Nets': 'BKN', 
            'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Dallas Mavericks': 'DAL', 'Detroit Pistons': 'DET', 'Houston Rockets': 'HOU', 
            'Indiana Pacers': 'IND', 'Milwaukee Bucks': 'MIL', 'New York Knicks': 'NYK', 'Sacramento Kings': 'SAC', 'Atlanta Hawks': 'ATL', 
            'Boston Celtics': 'BOS', 'Washington Wizards': 'WAS'}
    df["Team"] = df["Team"].str.replace("*", "")
    df["Team"] = df['Team'].map(dmap)
    return df[df['Team'] == home], df[df['Team'] == away]

#difference between key stats between the two teams 
def compute_differential_features(home_stats, opp_stats):
    # Only want these
    columns = ['3PA', '3P%','FTA', 'FT%', 'ORB', 'DRB','STL', 'BLK', 'TOV']
    diff_dict = {}
    for col in columns:
        diff_dict[f'{col}_diff'] = [home_stats.iloc[0][col] - opp_stats.iloc[0][col]]
    diff = pd.DataFrame(diff_dict)
    diff = rename_columns(diff)
    return diff



# ----- Model Training and Prediction -----
def train_model():
    """
    Trains a RandomForestClassifier on the DataFrame.
    Returns the trained classifier.
    """
    games = pd.read_csv(r"C:\Users\aaron\OneDrive\NBAstats\2024-games.csv")
    games['Win_home'] = games['wl_home'].apply(lambda x: 1 if x == 'W' else 0)
    games['Win_away'] = 0
    columns = ['fg3a', 'fg3_pct','fta', 'ft_pct', 'oreb', 'dreb','stl', 'blk', 'tov','Win']
    diff_dict = {f'{col}_diff': [] for col in columns}
    for index in games.index:
        for col in columns:
            diff_dict[f'{col}_diff'].append(games.loc[index][f'{col}_home'] - games.loc[index][f'{col}_away'])
    diff = pd.DataFrame(diff_dict)
    diff.dropna(inplace=True)

    feature_cols = [ 'fg3a_diff', 'fg3_pct_diff', 'fta_diff',
       'ft_pct_diff', 'oreb_diff', 'dreb_diff', 'stl_diff', 'blk_diff',
       'tov_diff']
    X = diff[feature_cols]
    y = diff['Win_diff']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Training Model Accuracy:", accuracy)
    return model

def rename_columns(df):
    new_columns = {
        'FGA_diff': 'fga_diff',
        'FG%_diff': 'fg_pct_diff',
        '3PA_diff': 'fg3a_diff',
        '3P%_diff': 'fg3_pct_diff',
        'FTA_diff': 'fta_diff',
        'FT%_diff': 'ft_pct_diff',
        'ORB_diff': 'oreb_diff',
        'DRB_diff': 'dreb_diff',
        'STL_diff': 'stl_diff',
        'BLK_diff': 'blk_diff',
        'TOV_diff': 'tov_diff'
    }
    df.rename(columns=new_columns, inplace=True)
    return df


def predict_win_probability(model, diff_pd):
    proba = model.predict_proba(diff_pd)[0]  # returns array like [prob_loss, prob_win]
    # Return probability that home team wins (class 1)
    return proba[1]

def main():
    df = pd.read_csv(r"C:/Users/aaron/OneDrive/NBAstats/teamPerGame.csv")
    cols = ['Rk', 'TRB', 'FG', '3P', 'FT', 'AST', 'PF', 'PTS', '2P', '2PA', '2P%']
    df.drop(cols, axis = 1, inplace=True)
    home_team = 'LAL'
    away_team = 'BOS'
    
    # Get stats for the two teams
    home,away = get_stats(df, home_team, away_team)
    
    diff = compute_differential_features(home, away)

    # Compute difference between key values features: (Home - Opponent)
    print("Differential Features:", diff)
    
    # Train the Random Forest classifier using past years data
    print("Training model on data...")
    model = train_model()
    
    # Predict win probability for the upcoming game (using current season stats)
    win_probability = predict_win_probability(model, diff)
    print(f"Predicted win probability for the home team {home_team}: {win_probability:.3f}")

main()
