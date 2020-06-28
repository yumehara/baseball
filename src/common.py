# coding:utf-8
import pandas as pd
import lightgbm as lgb

TRAIN_PITCH = '../data/train_pitch.f'
TEST_PITCH = '../data/test_pitch.f'

TRAIN_PLAYER = '../data/train_player.f'
TEST_PLAYER = '../data/test_player.f'

# Preprocess_player_2017.py
PLAYER_PIT_2017 = '../intermediate/player/pit_2017_{}.f'
PLAYER_CAT_2017 = '../intermediate/player/cat_2017_{}.f'
PLAYER_BAT_2017 = '../intermediate/player/bat_2017_{}.f'

# Preprocess_player
ALLPITCHER = '../intermediate/player/all_pitcher_{}.f'
ALLCATCHER = '../intermediate/player/all_catcher_{}.f'
ALLPLAYER = '../intermediate/player/all_player_{}.f'

# Preprocess_ball.py
BALL_2017 = '../intermediate/pitch/ball_2017.f'

# Preprocess_pitch.py
ALL_PITCH = '../intermediate/pitch/all_pitch.f'

# Preprocess_All.py
PREDICT_BALL = '../intermediate/{}/ball_predict_{}_{}.f'
PREDICT_COURSE = '../intermediate/{}/course_predict_{}_{}.f'

ALL_MERGE = '../intermediate/{}/all_merge_{}_{}.f'
ALL_MERGE_SUB = '../intermediate/{}/all_merge_{}_{}_sub.f'

OUTPUT_PATH = '../intermediate/{}'
SUBMIT_PATH = '../submit/{}'

# Predit_Ball.py
SUBMIT_BALL_CSV = '../submit/{}/ball_{}.csv'
SUBMIT_BALL_F = '../submit/{}/ball_{}_{}.f'
SUBMIT_BALL_SUB_CSV = '../submit/{}/ball_{}_sub.csv'
FI_BALL_F = '../submit/{}/ball_fi_{}.f'
FI_BALL_SUB_F = '../submit/{}/ball_fi_{}_sub.f'

# Predit_Course.py
SUBMIT_COURSE_CSV = '../submit/{}/course_{}.csv'
SUBMIT_COURSE_F = '../submit/{}/course_{}_{}.f'
SUBMIT_COURSE_SUB_CSV = '../submit/{}/course_{}_sub.csv'
FI_COURSE_F = '../submit/{}/course_fi_{}.f'
FI_COURSE_SUB_F = '../submit/{}/course_fi_{}_sub.f'

# ensemble.py
SUBMIT_BALL_ENSMBL_CSV = '../submit/{}/ball_ensmbl_{}_{}.csv'
SUBMIT_COURSE_ENSMBL_CSV = '../submit/{}/course_ensmbl_{}_{}.csv'

# log
LOG_SUBMIT = '../submit/{}/log.txt'

# 分割数
DIVIDE_NUM = 4
DIVIDE_1 = 60860
DIVIDE_2 = 120081
DIVIDE_3 = 180856
DIVIDE_4 = 257116

# DIVIDE_NUM = 5
# DIVIDE_1 = 51506
# DIVIDE_2 = 100844
# DIVIDE_3 = 150063
# DIVIDE_4 = 200447

# DIVIDE_NUM = 3
# DIVIDE_1 = 80792
# DIVIDE_2 = 160391
# DIVIDE_3 = 257116
# DIVIDE_4 = 257116

def divide_period_query_pre(sample_No):
    if sample_No == 1:
        return 'index <= {}'.format(DIVIDE_1)
    elif sample_No == 2:
        return 'index > {} & index <= {}'.format(DIVIDE_1, DIVIDE_2)
    elif sample_No == 3:
        return 'index > {} & index <= {}'.format(DIVIDE_2, DIVIDE_3)
    elif sample_No == 4:
        return 'index > {} & index <= {}'.format(DIVIDE_3, DIVIDE_4)
    # elif sample_No == 5:
    #     return 'index > {}'.format(DIVIDE_4)
    else:
        raise Exception('index error')

def divide_period_query_train(sample_No):
    if sample_No == 1:
        return 'index > {}'.format(DIVIDE_1)
    elif sample_No == 2:
        return 'index <= {} | index > {}'.format(DIVIDE_1, DIVIDE_2)
    elif sample_No == 3:
        return 'index <= {} | index > {}'.format(DIVIDE_2, DIVIDE_3)
    elif sample_No == 4:
        return 'index <= {} | index > {}'.format(DIVIDE_3, DIVIDE_4)
    # elif sample_No == 5:
    #     return 'index <= {}'.format(DIVIDE_4)
    else:
        raise Exception('index error')

def lightgbm_cv(lgb_param, lgb_train, num_round):
    cv_results = lgb.cv(lgb_param, lgb_train,
                    num_boost_round=num_round,
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
    df_feature_importance.sort_values('feat_imp', inplace=True, ascending=False)
    df_feature_importance.reset_index(inplace=True, drop=True)
    return df_feature_importance

def write_log(model_No, content):
    with open(LOG_SUBMIT.format(model_No), mode='a') as f:
        f.write(content)
        f.write('\n')