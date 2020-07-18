# coding:utf-8
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import feather
import common
import time

def preprocess(model_No, sample_No, use_sub_model):
    
    ALL_MERGE = common.ALL_MERGE.format(model_No, model_No, sample_No)
    
    all_pitch = pd.read_feather(ALL_MERGE)
    all_pitch = all_pitch.query(common.divide_period_query_train(sample_No))
    print(ALL_MERGE, all_pitch.shape)
    
    # sub-modelを使用するとき
    if use_sub_model:
        OUT_SUBMODEL = common.ALL_MERGE_SUB.format(model_No, model_No, sample_No)
        sub_model = pd.read_feather(OUT_SUBMODEL)
        print(OUT_SUBMODEL, sub_model.shape)
        all_pitch.reset_index(drop=True, inplace=True)
        all_pitch = pd.concat([all_pitch, sub_model], axis=1)

    print('all_pitch', all_pitch.shape)

    # train
    train = all_pitch.dropna(subset=['course'])
    train = train.query(common.divide_period_query_train(sample_No))
    print('train', train.shape)

    # test
    test = all_pitch[all_pitch['course'].isnull()]
    print('test', test.shape)

    del all_pitch
    gc.collect()

    train_d = train.drop([
        'No', 
        'course', 
        'ball'
    ], axis=1)

    test_d = test.drop([
        'No', 
        'course', 
        'ball'
    ], axis=1)

    return train_d, test_d, train['ball']

def tuning(model_No, use_sub_model, boosting, metric):
    
    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    sample_No = 1
    train_d, _, train_y = preprocess(model_No, sample_No, use_sub_model)

    filename = '../submit/{}/tuning_ball_{}_{}.log'.format(model_No, boosting, metric)
    common.tuning(train_d, train_y, 8, boosting, metric, filename)


def train_predict(model_No, use_sub_model, boosting, metric, sub_str):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)
    
    start = time.time()
    best_cv = []
    
    for sample_No in range(1, common.DIVIDE_NUM+1):
        
        SUBMIT = common.SUBMIT_BALL_CSV.format(model_No, model_No, sub_str)
        FI_RESULT = common.FI_BALL_F.format(model_No, sample_No, sub_str)
        SUBMIT_F = common.SUBMIT_BALL_F.format(model_No, model_No, sample_No)

        train_d, test_d, train_y = preprocess(model_No, sample_No, use_sub_model)

        lgb_param_gbdt = {
            'objective' : 'multiclass',
            'boosting_type': 'gbdt',
            'metric' : metric,
            'num_class' : 8,
            'seed' : 0,
            'learning_rate' : 0.1,
            'lambda_l1': 6.9923570049658075, 
            'lambda_l2': 0.002378623984798833, 
            'num_leaves': 18, 
            'feature_fraction': 0.45199999999999996, 
            'bagging_fraction': 0.9799724836460725, 
            'bagging_freq': 4, 
            'min_child_samples': 20
        }
        lgb_param_dart = {
            'objective' : 'multiclass',
            'boosting_type': 'dart',
            'metric' : metric,
            'num_class' : 8,
            'seed' : 0,
            'learning_rate' : 0.1,
            'lambda_l1': 3.2650173236383515, 
            'lambda_l2': 0.0006692176426537234, 
            'num_leaves': 39, 
            'feature_fraction': 0.552, 
            'bagging_fraction': 1.0, 
            'bagging_freq': 0, 
            'min_child_samples': 50
        }
        is_cv = True
        if boosting == common.GBDT:
            lgb_param = lgb_param_gbdt
            iter_num = 10000
        else:
            lgb_param = lgb_param_dart
            if metric == common.M_LOGLOSS:
                iter_num = 1400
            else:
                iter_num = 700
            if sample_No != 1:
                is_cv = False

        t1 = time.time()

        lgb_train = lgb.Dataset(train_d, train_y)
        # cross-varidation
        if is_cv:
            cv, best_iter = common.lightgbm_cv(lgb_param, lgb_train, iter_num, metric)
            best_cv.append(cv)

        t2 = time.time()
        print('lgb.cv: {} [s]'.format(t2 - t1))

        # train
        lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=best_iter)

        t3 = time.time()
        print('lgb.train: {} [s]'.format(t3 - t2))

        # feature importance
        fi = common.feature_importance(lgb_model)
        fi.to_feather(FI_RESULT)

        # predict
        predict = lgb_model.predict(test_d, num_iteration = lgb_model.best_iteration)

        t4 = time.time()
        print('lgb.predict: {} [s]'.format(t4 - t3))

        # result
        submit = pd.DataFrame(predict)
        submit.reset_index(inplace=True)
        print(submit.shape)

        # output feather
        submit_f = submit.drop(columns=['index'])
        submit_f.rename(columns={
                0: 'predict_straight', 1: 'predict_curve', 2: 'predict_slider', 3: 'predict_shoot',
                4: 'predict_fork', 5: 'predict_changeup', 6: 'predict_sinker', 7: 'predict_cutball'
            }, inplace=True)
        submit_f.to_feather(SUBMIT_F)
        print(SUBMIT_F, submit_f.shape)

    column_cnt = len(train_d.columns)

    # 結果まとめ
    result = common.SUBMIT_BALL_F.format(model_No, model_No, 1)
    print(result)
    df = pd.read_feather(result)
    columns = ['predict_straight', 'predict_curve', 'predict_slider', 'predict_shoot', 
               'predict_fork', 'predict_changeup', 'predict_sinker', 'predict_cutball']

    for i in range(2, common.DIVIDE_NUM+1):
        result = common.SUBMIT_BALL_F.format(model_No, model_No, i)
        print(result)
        temp = pd.read_feather(result)
        for c in columns:
            df[c] = df[c] + temp[c]

    for c in columns:
        df[c] = df[c] / common.DIVIDE_NUM
    
    cv_ave = 0
    for cv in best_cv:
        print('CV = {}'.format(cv))
        cv_ave = cv_ave + cv
    
    if len(best_cv) > 0:
        cv_ave = cv_ave / len(best_cv)
        print('CV(ave) = {}'.format(cv_ave))

    # 出力
    df = df.reset_index()
    df.to_csv(SUBMIT, header=False, index=False)
    print(SUBMIT)

    end = time.time()
    print('Predict_Ball: {} [s]'.format(end - start))

    signate_command = 'signate submit --competition-id=275 ./{} --note {}_{}_feat={}_cv={}'.format(SUBMIT, boosting, metric, column_cnt, cv_ave)
    common.write_log(model_No, signate_command)