# coding:utf-8
import pandas as pd
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_tune
import time
import datetime
from sklearn.model_selection import train_test_split

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
# PREDICT_BALL = '../intermediate/{}/ball_predict_{}_{}.f'
# PREDICT_COURSE = '../intermediate/{}/course_predict_{}_{}.f'

ALL_MERGE = '../intermediate/{}/all_merge_{}.f'
ALL_MERGE_SUB = '../intermediate/{}/all_merge_{}_sub.f'

OUTPUT_PATH = '../intermediate/{}'
SUBMIT_PATH = '../submit/{}'

# Predit_Ball.py
SUBMIT_BALL_CSV = '../submit/{}/ball_{}_{}.csv'
SUBMIT_BALL_F = '../submit/{}/ball_{}_{}.f'
FI_BALL_F = '../submit/{}/ball_fi_{}_{}.f'

# Predit_Course.py
SUBMIT_COURSE_CSV = '../submit/{}/course_{}_{}.csv'
SUBMIT_COURSE_F = '../submit/{}/course_{}_{}.f'
FI_COURSE_F = '../submit/{}/course_fi_{}_{}.f'
COURSE_TRAIN = '../submit/{}/course_train_{}_{}.f'

# Predit_Lastball.py
LAST_BALL_SUB = '../submit/{}/lastball_sub_{}.f'

# ensemble.py
SUBMIT_BALL_ENSMBL_CSV = '../submit/{}/ball_{}_{}_{}.csv'
SUBMIT_COURSE_ENSMBL_CSV = '../submit/{}/course_{}_{}_{}.csv'

# log
LOG_SUBMIT = '../log/log_{}.txt'

# 分割数
DIVIDE_NUM = 0
DIVIDE_1 = 0
DIVIDE_2 = 0
DIVIDE_3 = 0
DIVIDE_4 = 0
DIVIDE_5 = 257116
SEED = 0

LEARNING_RATE = 0.05

def set_divide_num(num):
    global DIVIDE_NUM
    global DIVIDE_1
    global DIVIDE_2
    global DIVIDE_3
    global DIVIDE_4
    global SEED

    if num == 4:
        DIVIDE_NUM = 4
        DIVIDE_1 = 60860
        DIVIDE_2 = 120081
        DIVIDE_3 = 180856
        DIVIDE_4 = 257116
        SEED = 0
    elif num == 5:
        DIVIDE_NUM = 5
        DIVIDE_1 = 51506
        DIVIDE_2 = 100844
        DIVIDE_3 = 150063
        DIVIDE_4 = 200447
        SEED = 99
    elif num == 3:
        DIVIDE_NUM = 3
        DIVIDE_1 = 80792
        DIVIDE_2 = 160391
        DIVIDE_3 = 257116
        DIVIDE_4 = 257116
        SEED = 999
    else:
        raise Exception('divide-num error')

# boosting
GBDT = 'gbdt'
DART = 'dart'

# metric
M_LOGLOSS = 'multi_logloss'
M_ERROR = 'multi_error'

def divide_period_query_pre(sample_No):
    if sample_No == 1:
        return 'index <= {}'.format(DIVIDE_1)
    elif sample_No == 2:
        return 'index > {} & index <= {}'.format(DIVIDE_1, DIVIDE_2)
    elif sample_No == 3:
        return 'index > {} & index <= {}'.format(DIVIDE_2, DIVIDE_3)
    elif sample_No == 4:
        return 'index > {} & index <= {}'.format(DIVIDE_3, DIVIDE_4)
    elif sample_No == 5:
        return 'index > {}'.format(DIVIDE_4)
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
    elif sample_No == 5:
        return 'index <= {} | index > {}'.format(DIVIDE_4, DIVIDE_5)
    else:
        raise Exception('index error')

def lightgbm_cv(lgb_param, lgb_train, num_round, metric, stratified=True):
    cv_results = lgb.cv(lgb_param, lgb_train,
                    num_boost_round=num_round,
                    early_stopping_rounds=100,
                    verbose_eval=100,
                    nfold=4,
                    stratified=stratified)

    metric_mean = metric + '-mean'
    num_boost_round = len(cv_results[metric_mean])
    print('Best num_boost_round:', num_boost_round)
    best_cv_score = cv_results[metric_mean][-1]
    print('Best CV score:', best_cv_score)
    best_iter = int(num_boost_round)
    return best_cv_score, best_iter

def feature_importance(lgb_model):
    fi = lgb_model.feature_importance()
    fn = lgb_model.feature_name()
    df_feature_importance = pd.DataFrame({'feat_name':fn, 'feat_imp':fi})
    df_feature_importance.sort_values('feat_imp', inplace=True, ascending=False)
    df_feature_importance.reset_index(inplace=True, drop=True)
    return df_feature_importance

JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')

def write_log(model_No, content):
    write_text(LOG_SUBMIT.format(model_No), content)

def write_text(filename, content):
    with open(filename, mode='a') as f:
        text = '[{}] {} \n'.format(datetime.datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S'), content)
        f.write(text)
        print(content)

def tuning(train_x, train_y, num_class, boosting, metric, num_round, logfile):
    
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, random_state=0)
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)

    lgb_train = lgb_tune.Dataset(X_train, y_train)
    lgb_eval = lgb_tune.Dataset(X_test, y_test, reference=lgb_train)

    lgb_param = {
        'objective' : 'multiclass',
        'boosting_type': boosting,
        'metric' : metric,
        'num_class' : num_class,
    }
    best_params, tuning_history = dict(), list()
    lgb_model = lgb_tune.train(lgb_param, lgb_train,
                        valid_sets=lgb_eval,
                        verbose_eval=0,
                        num_boost_round=num_round,
                        best_params=best_params,
                        tuning_history=tuning_history)
    
    end = time.time()
    tuning_time = 'tuning_Course: {} [s]'.format(end - start)
    write_text(logfile, tuning_time)
    write_text(logfile, 'Best Params:')
    write_text(logfile, best_params)