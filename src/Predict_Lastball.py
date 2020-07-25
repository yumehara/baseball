# coding:utf-8
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import feather
import common
import time


def preprocess(model_No, sample_No):

    ALL_MERGE = common.ALL_MERGE.format(model_No, sample_No)
    all_pitch = pd.read_feather(ALL_MERGE)
    all_pitch = all_pitch.query(common.divide_period_query_train(sample_No))
    print(ALL_MERGE, all_pitch.shape)

    # train
    train = all_pitch.dropna(subset=['course'])
    print('train', train.shape)

    # test
    test = all_pitch[all_pitch['course'].isnull()]
    print('test', test.shape)

    del all_pitch
    gc.collect()

    train_d = train.drop([
        'No', 
        'course', 
        'ball',
        'last_ball'
    ], axis=1)

    test_d = test.drop([
        'No', 
        'course', 
        'ball',
        'last_ball'
    ], axis=1)

    return train_d, test_d, train['last_ball']


def train_predict(model_No, boosting, metric):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    start = time.time()
    best_cv = []
    common.write_log(model_No, 'Lastball.train_predict {}, {}'.format(boosting, metric))
    
    for sample_No in range(1, common.DIVIDE_NUM+1):

        OUT_SUBMODEL = common.LAST_BALL_SUB.format(model_No, sample_No)
        
        train_d, test_d, train_y = preprocess(model_No, sample_No)

        lgb_param = {
            'objective' : 'regression',
            'boosting_type': boosting,
            'metric' : metric,
            'seed' : 0,
            'learning_rate' : 0.1,
        }
        is_cv = False
        if sample_No == 1:
            is_cv = True
            iter_num = 20000
        
        t1 = time.time()

        lgb_train = lgb.Dataset(train_d, train_y)
        # cross-varidation
        if is_cv:
            cv, best_iter = common.lightgbm_cv(lgb_param, lgb_train, iter_num, metric)
            best_cv.append(cv)
            common.write_log(model_No, 'CV({}) = {}, best_iter = {}'.format(sample_No, cv, best_iter))

        t2 = time.time()
        print('lgb.cv: {} [s]'.format(t2 - t1))

        # train
        lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=best_iter)

        t3 = time.time()
        print('lgb.train: {} [s]'.format(t3 - t2))

        # predict
        predict = lgb_model.predict(test_d, num_iteration = lgb_model.best_iteration)

        t4 = time.time()
        print('lgb.predict: {} [s]'.format(t4 - t3))

        # result
        submit = pd.DataFrame(predict)
        submit.reset_index(inplace=True)
        print(submit.shape)

        # train predict
        train_predict = lgb_model.predict(train_d, num_iteration = lgb_model.best_iteration)
        df_train_predict = pd.DataFrame(train_predict).reset_index()
        submodel = pd.concat([df_train_predict, submit], ignore_index=True)
        submodel.drop(columns=['index'], inplace=True)
        submodel.rename(columns={0: 'predict_lastball'}, inplace=True)
        submodel.to_feather(OUT_SUBMODEL)
        print(OUT_SUBMODEL, submodel.shape)
    
    cv_ave = 0
    for cv in best_cv:
        print('CV = {}'.format(cv))
        cv_ave = cv_ave + cv
    
    if len(best_cv) > 0:
        cv_ave = cv_ave / len(best_cv)
        common.write_log(model_No, 'CV(ave) = {}'.format(cv_ave))

    end = time.time()
    print('Predict_Lastball: {} [s]'.format(end - start))
