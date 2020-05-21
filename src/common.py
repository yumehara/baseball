# coding:utf-8
import pandas as pd
import lightgbm as lgb

TRAIN_PITCH = '../data/train_pitch.f'
TEST_PITCH = '../data/test_pitch.f'

TRAIN_PLAYER = '../data/train_player.f'
TEST_PLAYER = '../data/test_player.f'

# Preprocess_player_2017.py
PLAYER_PIT_2017 = '../intermediate/player/pit_2017_7_{}.f'
PLAYER_BAT_2017 = '../intermediate/player/bat_2017_7_{}.f'

# Preprocess_player
ALLPITCHER = '../intermediate/player/all_pitcher_16_{}.f'
ALLPLAYER = '../intermediate/player/all_player_16_{}.f'

# Preprocess_ball.py
BALL_2017 = '../intermediate/pitch/ball_2017_5.f'

# Preprocess_pitch.py
ALL_PITCH = '../intermediate/pitch/all_pitch_14.f'

# Preprocess_All.py
PREDICT_BALL = '../intermediate/{}/ball_predict_{}_{}.f'
PREDICT_COURSE = '../intermediate/{}/course_predict_{}_{}.f'

ALL_MERGE = '../intermediate/{}/all_merge_{}_{}.f'
ALL_MERGE_SUB = '../intermediate/{}/all_merge_{}_{}_sub.f'

OUTPUT_PATH = '../intermediate/{}'

# Predit_Ball.py
SUBMIT_BALL_CSV = '../submit/{}/ball_{}.csv'
SUBMIT_BALL_F = '../submit/{}/ball_{}_{}.f'
SUBMIT_BALL_SUB_CSV = '../submit/{}/ball_{}_sub.csv'


# 分割数
DIVIDE_NUM = 4
DIVIDE_1 = 60000
DIVIDE_2 = 120000
DIVIDE_3 = 180000

def divide_period_query_pre(sample_No):
    if sample_No == 1:
        return 'index <= {}'.format(DIVIDE_1)
    elif sample_No == 2:
        return 'index > {} & index <= {}'.format(DIVIDE_1, DIVIDE_2)
    elif sample_No == 3:
        return 'index > {} & index <= {}'.format(DIVIDE_2, DIVIDE_3)
    elif sample_No == 4:
        return 'index > {}'.format(DIVIDE_3)

def divide_period_query_train(sample_No):
    if sample_No == 1:
        return 'index > {}'.format(DIVIDE_1)
    elif sample_No == 2:
        return 'index <= {} & index > {}'.format(DIVIDE_1, DIVIDE_2)
    elif sample_No == 3:
        return 'index <= {} & index > {}'.format(DIVIDE_2, DIVIDE_3)
    elif sample_No == 4:
        return 'index <= {}'.format(DIVIDE_3)

def lightgbm_cv(lgb_param, lgb_train):
    cv_results = lgb.cv(lgb_param, lgb_train,
                    num_boost_round=15000,
                    early_stopping_rounds=100,
                    verbose_eval=100,
                    nfold=4)

    num_boost_round = len(cv_results['multi_logloss-mean'])
    print('Best num_boost_round:', num_boost_round)
    best_cv_score = cv_results['multi_logloss-mean'][-1]
    print('Best CV score:', best_cv_score)
    best_iter = int(num_boost_round * 1.1)
    return best_cv_score, best_iter

def feature_importance(lgb_model):
    fi = lgb_model.feature_importance()
    fn = lgb_model.feature_name()
    df_feature_importance = pd.DataFrame({'feat_name':fn, 'feat_imp':fi})
    df_feature_importance.sort_values('feat_imp', inplace=True)
    return df_feature_importance