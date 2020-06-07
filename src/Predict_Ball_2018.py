# coding:utf-8
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import feather
import common

def train_predict(model_No, use_sub_model):
    
    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)
    
    best_cv = []
    divide_num = common.get_divide_num(False)
    for sample_No in range(1, divide_num):
        
        # if use_sub_model:
        #     ALL_MERGE = common.ALL_MERGE_SUB.format(model_No, model_No, sample_No)
        #     SUBMIT = common.SUBMIT_BALL2_SUB_CSV.format(model_No, model_No)
        # else:
        ALL_MERGE = common.ALL_MERGE_2018.format(model_No, model_No, sample_No)
        SUBMIT = common.SUBMIT_BALL2_CSV.format(model_No, model_No)

        SUBMIT_F = common.SUBMIT_BALL2_F.format(model_No, model_No, sample_No)
        # OUT_SUBMODEL = common.PREDICT_BALL.format(model_No, model_No, sample_No)

        all_pitch = pd.read_feather(ALL_MERGE)
        '''
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
        '''
        column_cnt = len(all_pitch.columns)
        print(all_pitch.shape)

        # train
        train = all_pitch.dropna(subset=['course'])
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

        lgb_param = {
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

        lgb_train = lgb.Dataset(train_d, train['ball'])
        # cross-varidation
        cv, best_iter = common.lightgbm_cv(lgb_param, lgb_train)
        best_cv.append(cv)
        # train
        lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=best_iter)

        # fi = common.feature_importance(lgb_model).tail(30)

        # predict
        predict = lgb_model.predict(test_d, num_iteration = lgb_model.best_iteration)

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
        '''
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
        '''
    column_cnt = len(train_d.columns)
    
    # 結果まとめ
    result = common.PREDICT_BALL_2018.format(model_No)
    print(result)
    df = pd.read_feather(result)
    columns = ['predict_straight', 'predict_curve', 'predict_slider', 'predict_shoot', 
               'predict_fork', 'predict_changeup', 'predict_sinker', 'predict_cutball']

    sample_no = 1
    for div in common.DIVIDE_TEST:
        result = common.SUBMIT_BALL2_F.format(model_No, model_No, sample_no)
        print(result)
        sample_no = sample_no + 1
        
        temp = pd.read_feather(result)
        for c in columns:
            temp.loc[temp.index <= div, c] = 0
            df[c] = df[c] + temp[c]

    count = 1
    df['count'] = count
    for div in common.DIVIDE_TEST:
        count = count + 1
        df.loc[df.index > div, 'count'] = count
    
    for c in columns:
        df[c] = df[c] / df['count']

    df.drop(columns=['count'], inplace=True)

    cv_ave = 0
    for cv in best_cv:
        print('CV = {}'.format(cv))
        cv_ave = cv_ave + cv
    
    cv_ave = cv_ave / common.DIVIDE_NUM
    print('CV(ave) = {}'.format(cv_ave))

    # 出力
    df = df.reset_index()
    df.to_csv(SUBMIT, header=False, index=False)
    print(SUBMIT)

    print('signate submit --competition-id=275 ./{} --note feat={}_cv={}'.format(SUBMIT, column_cnt, cv_ave))
    