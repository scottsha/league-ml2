#### GBM1 No extra attributes ####
# Author: Albert Bush
# Date: 24 March 2018
# GBM to test how good individual win rates are at predicting outcomes.

# Load packages
library(gbm)
library(data.table)
library(bit64)
library(dplyr)
library(stringr)

#### Load training set ####
setwd("C:/Users/Albert/Desktop/Programming/league-ml2/src/models")
source('../model_evaluation/model_performance_functions.R')
ALL_MATCHES = '../../data/raw/matches_230k_3_24_2018.csv'
all_data = fread(ALL_MATCHES, data.table = FALSE)
WIN_RATES = '../../data/interim/indiv_win_rates_3_24_2018.csv'
RESULTS_OUTPUT = '../../data/model_performance/gbm1_indiv_winrates_only_230k.csv'
PARAMS_LM_OUT = '../../data/model_performance/gbm1_indiv_winrates_only_230k_parameter_sig.txt'

#### Preprocess to prep for input to GBM ####
win_rates = fread(WIN_RATES, data.table = FALSE)
names(win_rates)[1] = 'champs'

team_lane_positions = c('100_TOP_SOLO',
                        '100_JUNGLE_NONE',
                        '100_MIDDLE_SOLO',
                        '100_BOTTOM_DUO_CARRY',
                        '100_BOTTOM_DUO_SUPPORT',
                        '200_TOP_SOLO',
                        '200_JUNGLE_NONE',
                        '200_MIDDLE_SOLO',
                        '200_BOTTOM_DUO_CARRY',
                        '200_BOTTOM_DUO_SUPPORT')

for(tlp in team_lane_positions) {
  names(win_rates)[1] = tlp
  win_rate_col = paste0(substr(tlp,5,100),'_win_rate')
  games_played_col = paste0(substr(tlp,5,100),'_games_played')
  all_data = left_join(all_data,
                       win_rates[c(tlp, win_rate_col, games_played_col)],
                       by = tlp)
  names(all_data)[names(all_data) == win_rate_col] = paste0(tlp,'_win_rate')
  names(all_data)[names(all_data) == games_played_col] = paste0(tlp,'_games_played')
}



# Specify attributes to use in model, slim the training set,
# rename attributes (100_ is not valid), and convert to factors
attributes_for_modeling = c('100_TOP_SOLO_win_rate',
                            '100_JUNGLE_NONE_win_rate',
                            '100_MIDDLE_SOLO_win_rate',
                            '100_BOTTOM_DUO_CARRY_win_rate',
                            '100_BOTTOM_DUO_SUPPORT_win_rate',
                            '200_TOP_SOLO_win_rate',
                            '200_JUNGLE_NONE_win_rate',
                            '200_MIDDLE_SOLO_win_rate',
                            '200_BOTTOM_DUO_CARRY_win_rate',
                            '200_BOTTOM_DUO_SUPPORT_win_rate',
                            '100_TOP_SOLO_games_played',
                            '100_JUNGLE_NONE_games_played',
                            '100_MIDDLE_SOLO_games_played',
                            '100_BOTTOM_DUO_CARRY_games_played',
                            '100_BOTTOM_DUO_SUPPORT_games_played',
                            '200_TOP_SOLO_games_played',
                            '200_JUNGLE_NONE_games_played',
                            '200_MIDDLE_SOLO_games_played',
                            '200_BOTTOM_DUO_CARRY_games_played',
                            '200_BOTTOM_DUO_SUPPORT_games_played')


all_data = all_data[ ,c('team_100_win', attributes_for_modeling)]

names(all_data) = str_replace_all(names(all_data), '100', 'blue')
names(all_data) = str_replace_all(names(all_data), '200', 'red')

#all_data = rename(all_data, blue_top = '100_TOP_SOLO',
#                  blue_jg = '100_JUNGLE_NONE',
                  # blue_mid = '100_MIDDLE_SOLO',
                  # blue_adc = '100_BOTTOM_DUO_CARRY',
                  # blue_supp = '100_BOTTOM_DUO_SUPPORT',
                  # red_top = '200_TOP_SOLO',
                  # red_jg = '200_JUNGLE_NONE',
                  # red_mid = '200_MIDDLE_SOLO',
                  # red_adc = '200_BOTTOM_DUO_CARRY',
                  # red_supp = '200_BOTTOM_DUO_SUPPORT')

#attributes_for_modeling = names(all_data)[2:length(all_data)]

# Create formula and convert columns to factors
basic_formula = 'team_blue_win ~ '
for(attr in names(all_data)) {
  if(attr != 'team_blue_win') {
    basic_formula = paste0(basic_formula, attr, ' + ')
    all_data[[attr]] = as.factor(all_data[[attr]])
  }
}
set.seed(1)
train_recs = sample(1:nrow(all_data), .8 * nrow(all_data))
train = all_data[train_recs,]
validation = all_data[!(1:nrow(all_data) %in% train_recs),]

# Get rid of extra + at the end
basic_formula = substr(basic_formula, 1, nchar(basic_formula) - 3)
basic_formula = as.formula(basic_formula)

#interaction_depth = c(1,2,4,6,8,10)
#shrinkage_rate = c(.1, .05, .01)
#bag_fraction = c(.3, .6, .9)
#cv_folds = c(2,5)
#num_trees = c(1000)

interaction_depth = c(5)
shrinkage_rate = c(.1)
bag_fraction = c(1.0)
cv_folds = c(5)
num_trees = c(500)


param_df = setNames(expand.grid(interaction_depth, shrinkage_rate, bag_fraction, cv_folds,num_trees),
                    c('interaction_depth','shrinkage_rate','bag_fraction','cv_folds','num_trees'))

results_df = setNames(data.frame(matrix(nrow = 0, ncol = 19)),
                      c('inter_dpth','shrink','bag_fr','cv_fo','n.trees','best.iter','ks_train','gini_train',
                        'ks_valid','gini_valid','precision_train','recall_train','precision_valid',
                        'recall_valid','auc_train','auc_valid','train_time','train_size','valid_size'))

for(i in 1:nrow(param_df)) {
  start_time = Sys.time()
  print(param_df[i,])
  #### Train model ####
  gbm1_noattr = gbm(basic_formula,
                    distribution = 'bernoulli',
                    data = train,
                    interaction.depth = param_df[i,'interaction_depth'],
                    shrinkage = param_df[i,'shrinkage_rate'],
                    bag.fraction = param_df[i,'bag_fraction'],
                    train.fraction = 1.0,
                    n.trees = param_df[i,'num_trees'],
                    verbose = TRUE,
                    cv.folds = param_df[i,'cv_folds'])
  
  #### Evaluate Model ####
  best.iter = gbm.perf(gbm1_noattr, method = 'cv')
  train_pred = predict.gbm(gbm1_noattr, train, best.iter)
  valid_pred = predict.gbm(gbm1_noattr, validation, best.iter)
  ks_gini_train = ks_gini(train$team_blue_win, train_pred)
  ks_gini_valid = ks_gini(validation$team_blue_win, valid_pred)
  precision_recall_train = precision_recall(train$team_blue_win, train_pred) 
  precision_valid = precision_recall(validation$team_blue_win, valid_pred)
  #auc_train = rank_comparison_auc(train$team_100_win, train_pred)
#  auc_valid = rank_comparison_auc(validation$team_blue_win, valid_pred)
  end_time = Sys.time()
  total_time = end_time - start_time
  cur_row = nrow(results_df)+1
  results_df[cur_row,] = c(as.numeric(param_df[i,]), 
                           best.iter, 
                           ks_gini_train[1:2], 
                           ks_gini_valid[1:2],
                           precision_recall_train,
                           precision_valid,
                           'NA',
                           'NA',
                           total_time,
                           nrow(train),
                           nrow(validation))
  print(results_df)
  # Write results as well as gbm parameters to a file with append = true to save a running tally of what ive done before
  write.table(results_df[cur_row,], RESULTS_OUTPUT, sep = ',', 
              quote = FALSE, row.names = FALSE, col.names = (i == 1), append = TRUE)
}
#write.csv(results_df, '../../data/model_performance/gbm1_noattributes_75k.csv', 
#          quote = FALSE, row.names = FALSE)

for(i in c('inter_dpth','shrink','bag_fr','cv_fo','n.trees','best.iter','ks_valid')) {
  results_df[[i]] = as.numeric(results_df[[i]])
}

lm_results_analysis = lm(ks_valid ~ inter_dpth + shrink +	bag_fr + cv_fo + n.trees + best.iter, results_df)
sink(PARAMS_LM_OUT)
print(summary(lm_results_analysis))
sink()
# higher bag fraction, lower inter depth, lower shrinkage