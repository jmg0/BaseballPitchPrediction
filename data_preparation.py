import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prep_data_single(input_file='./ScherzerExampleData.csv', input_df=None):
    if input_file:
        df = pd.read_csv(input_file)
    else:
        df = input_df
    label_encoder = LabelEncoder()
    df['prev_pitch_type'] = label_encoder.fit_transform(df['pitch_type'])
    feature_vars = ['prev_pitch_type', 'release_speed', 'batter', 'balls', 'strikes', 'game_year',
                    'outs_when_up', 'inning', 'release_spin_rate', 'home_score', 'away_score']
    features, labels = df[feature_vars], df['prev_pitch_type']
    features, labels = features.to_numpy(), labels.to_numpy()
    features, labels = np.flip(features, axis=0), labels[::-1]
    features, labels = np.append(np.full((1, len(feature_vars)), fill_value=-1), features, axis=0), np.append(labels, np.array([-1]))
    return features, labels

def prep_data(input_file):
    df = pd.read_csv(input_file)
    pitchers = df['player_name'].unique()
    pitcher_data_dict = dict()
    for pitcher in pitchers:
        pitcher_data_dict[pitcher] = prep_data_single(input_file=None, input_df=df[df['player_name'] == pitcher])
    return pitcher_data_dict