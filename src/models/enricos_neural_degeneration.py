import pandas as pd
import numpy as np
import keras as kr
import keras.layers as dn
import keras.models as seq
from sklearn.utils import shuffle

match_alb = pd.read_csv('../../processed_match_data.csv')
match_alb = shuffle(match_alb).reset_index()
train_size = round(0.6*match_alb.shape[0])
vel_end = round(0.8*match_alb.shape[0])
trnd = match_alb[0:train_size].reset_index()
vel = match_alb[train_size:vel_end].reset_index()


champs = match_alb['100_TOP_SOLO'].unique()
np.append(champs, match_alb['100_JUNGLE_NONE'].unique())
np.append(champs, match_alb['100_MIDDLE_SOLO'].unique())
np.append(champs, match_alb['100_BOTTOM_DUO_CARRY'].unique())
np.append(champs, match_alb['100_BOTTOM_DUO_SUPPORT'].unique())
champs = np.unique(champs)
champ_num = dict([(bar,foo) for foo,bar in enumerate(champs)])
num_champ = dict([(foo,bar) for foo,bar in enumerate(champs)])
L=len(champ_num)

def rec_to_vec( row ):
    vec=np.zeros(L)
    vec[champ_num[row['100_TOP_SOLO']]] = 1
    vec[champ_num[row['100_JUNGLE_NONE']]] = 1
    vec[champ_num[row['100_MIDDLE_SOLO']]] = 1
    vec[champ_num[row['100_BOTTOM_DUO_CARRY']]] = 1
    vec[champ_num[row['100_BOTTOM_DUO_SUPPORT']]] = 1
    vec[champ_num[row['200_TOP_SOLO']]] = -1
    vec[champ_num[row['200_JUNGLE_NONE']]] = -1
    vec[champ_num[row['200_MIDDLE_SOLO']]] = -1
    vec[champ_num[row['200_BOTTOM_DUO_CARRY']]] = -1
    vec[champ_num[row['200_BOTTOM_DUO_SUPPORT']]] = -1
    return vec

train_d = np.zeros([train_size,L])
print(train_d.size)
for foo, row in trnd.iterrows():
    train_d[foo,:] = rec_to_vec(row)

valid_d = np.zeros([vel.shape[0],L])
for foo, row in vel.iterrows():
    valid_d[foo,:] = rec_to_vec(row)


model = seq.Sequential()
model.add(dn.Dense(70, activation='sigmoid', input_shape= (train_d.shape[1],) ))
model.add(dn.Dense(35, activation='sigmoid'))
model.add(dn.Dense(18, activation='sigmoid'))
model.add(dn.Dense(9, activation='sigmoid'))
model.add(dn.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_d, trnd['team_100_win'])

#------------------------------#
#CONCEPTUAL DISTINCTION
#-----------------------------#

train_p = model.predict(train_d)
valid_p = model.predict(valid_d)

def correct_prediction_rate(loss, score, threshold=0.5):
    """Calculate the percentage of games correctly predicted."""
    correct_blue_team_win = np.sum(np.logical_and(score >= threshold, loss == 1))
    correct_red_team_win = np.sum(np.logical_and(score < threshold, loss == 0))
    return (correct_blue_team_win + correct_red_team_win)/len(loss)

c1 = correct_prediction_rate(trnd['team_100_win'], train_p[:,0])
c2 = correct_prediction_rate(vel['team_100_win'], valid_p[:,0])
print(c1, c2)