# coding:utf-8
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import feather
import common
import time

def train_predict(model_No, use_sub_model, is_gbdt):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)
    
    start = time.time()
    best_cv = []
    
    for sample_No in range(1, common.DIVIDE_NUM+1):
        
        if use_sub_model:
            ALL_MERGE = common.ALL_MERGE_SUB.format(model_No, model_No, sample_No)
            SUBMIT = common.SUBMIT_BALL_SUB_CSV.format(model_No, model_No)
            FI_RESULT = common.FI_BALL_SUB_F.format(model_No, sample_No)
        else:
            ALL_MERGE = common.ALL_MERGE.format(model_No, model_No, sample_No)
            SUBMIT = common.SUBMIT_BALL_CSV.format(model_No, model_No)
            FI_RESULT = common.FI_BALL_F.format(model_No, sample_No)

        SUBMIT_F = common.SUBMIT_BALL_F.format(model_No, model_No, sample_No)
        OUT_SUBMODEL = common.PREDICT_BALL.format(model_No, model_No, sample_No)

        all_pitch = pd.read_feather(ALL_MERGE)
        
        # sub-modelを使用するとき
        if use_sub_model:
            all_pitch['predict_high_str'] = all_pitch['predict_0'] + all_pitch['predict_3'] + all_pitch['predict_6'] 
            all_pitch['predict_high_ball'] = all_pitch['predict_9'] + all_pitch['predict_10'] 
            all_pitch['predict_mid_str'] = all_pitch['predict_1'] + all_pitch['predict_4'] + all_pitch['predict_7'] 
            all_pitch['predict_low_str'] = all_pitch['predict_2'] + all_pitch['predict_5'] + all_pitch['predict_8'] 
            all_pitch['predict_low_ball'] = all_pitch['predict_11'] + all_pitch['predict_12'] 

            all_pitch['predict_left_str'] = all_pitch['predict_0'] + all_pitch['predict_1'] + all_pitch['predict_2'] 
            all_pitch['predict_left_ball'] = all_pitch['predict_9'] + all_pitch['predict_11'] 
            all_pitch['predict_center_str'] = all_pitch['predict_3'] + all_pitch['predict_4'] + all_pitch['predict_5'] 
            all_pitch['predict_right_str'] = all_pitch['predict_6'] + all_pitch['predict_7'] + all_pitch['predict_8'] 
            all_pitch['predict_right_ball'] = all_pitch['predict_10'] + all_pitch['predict_12']

            all_pitch.drop(columns=[
                # 'predict_straight', 'predict_curve', 'predict_slider', 'predict_shoot',
                # 'predict_fork', 'predict_changeup', 'predict_sinker', 'predict_cutball',
                'predict_0','predict_1','predict_2','predict_3','predict_4','predict_5','predict_6',
                'predict_7','predict_8','predict_9','predict_10','predict_11','predict_12'
            ], inplace=True)

        column_cnt = len(all_pitch.columns)
        print(all_pitch.shape)

        # train
        train = all_pitch.dropna(subset=['course'])
        train = train.query(common.divide_period_query_train(sample_No))
        print(train.shape)

        # test
        test = all_pitch[all_pitch['course'].isnull()]
        print(test.shape)

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

        lgb_param_gbdt = {
            'objective' : 'multiclass',
            'boosting_type': 'gbdt',
            'metric' : 'multi_logloss',
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
            'metric' : 'multi_logloss',
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
        if is_gbdt:
            lgb_param = lgb_param_gbdt
            iter_num = 10000
        else:
            lgb_param = lgb_param_dart
            iter_num = 1400
            if sample_No != 1:
                is_cv = False

        t1 = time.time()

        lgb_train = lgb.Dataset(train_d, train['ball'])
        # cross-varidation
        if is_cv:
            cv, best_iter = common.lightgbm_cv(lgb_param, lgb_train, iter_num)
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

        # コース予測で使用
        if not use_sub_model:
            train_predict = lgb_model.predict(train_d, num_iteration = lgb_model.best_iteration)

            df_train_predict = pd.DataFrame(train_predict).reset_index()
            submodel = pd.concat([df_train_predict, submit], ignore_index=True)
            submodel.drop(columns=['index'], inplace=True)
            submodel.rename(columns={
                0: 'predict_straight', 1: 'predict_curve', 2: 'predict_slider', 3: 'predict_shoot', 
                4: 'predict_fork', 5: 'predict_changeup', 6: 'predict_sinker', 7: 'predict_cutball'
            }, inplace=True)
            
            submodel.to_feather(OUT_SUBMODEL)
            print(OUT_SUBMODEL, submodel.shape)

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

    print('signate submit --competition-id=275 ./{} --note feat={}_cv={}'.format(SUBMIT, column_cnt, cv_ave))
    