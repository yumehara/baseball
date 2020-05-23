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
    for sample_No in range(1, common.DIVIDE_NUM+1):
        
        if use_sub_model:
            ALL_MERGE = common.ALL_MERGE_SUB.format(model_No, model_No, sample_No)
            SUBMIT = common.SUBMIT_COURSE_SUB_CSV.format(model_No, model_No)
        else:
            ALL_MERGE = common.ALL_MERGE.format(model_No, model_No, sample_No)
            SUBMIT = common.SUBMIT_COURSE_CSV.format(model_No, model_No)

        SUBMIT_F = common.SUBMIT_COURSE_F.format(model_No, model_No, sample_No)
        OUT_SUBMODEL = common.PREDICT_COURSE.format(model_No, model_No, sample_No)

        all_pitch = pd.read_feather(ALL_MERGE)

        # sub-modelを使用するとき
        if use_sub_model:
            all_pitch['predict_curve'] = all_pitch['predict_curve'] / all_pitch['predict_straight']
            all_pitch['predict_slider'] = all_pitch['predict_slider'] / all_pitch['predict_straight']
            all_pitch['predict_shoot'] = all_pitch['predict_shoot'] / all_pitch['predict_straight']
            all_pitch['predict_fork'] = all_pitch['predict_fork'] / all_pitch['predict_straight']
            all_pitch['predict_changeup'] = all_pitch['predict_changeup'] / all_pitch['predict_straight']
            all_pitch['predict_sinker'] = all_pitch['predict_sinker'] / all_pitch['predict_straight']
            all_pitch['predict_cutball'] = all_pitch['predict_cutball'] / all_pitch['predict_straight']
            all_pitch.drop(columns=[
                # 'predict_0', 'predict_1', 'predict_2', 'predict_3', 'predict_4', 'predict_5', 'predict_6',
                # 'predict_7', 'predict_8', 'predict_9', 'predict_10', 'predict_11', 'predict_12',
                'predict_straight'
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

        lgb_param = {
            'objective' : 'multiclass',
            'boosting_type': 'gbdt',
            'metric' : 'multi_logloss',
            'num_class' : 13,
            'seed' : 0,
            'learning_rate' : 0.01,
            'lambda_l1': 8.769293390201968, 
            'lambda_l2': 3.913949617576324e-05, 
            'num_leaves': 6, 
            'feature_fraction': 0.4, 
            'bagging_fraction': 0.8391111798378441, 
            'bagging_freq': 4, 
            'min_child_samples': 50
        }

        lgb_train = lgb.Dataset(train_d, train['course'])
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
            0: 'predict_0', 1: 'predict_1', 2: 'predict_2', 3: 'predict_3',
            4: 'predict_4', 5: 'predict_5', 6: 'predict_6', 7: 'predict_7',
            8: 'predict_8', 9: 'predict_9', 10: 'predict_10', 11: 'predict_11', 12: 'predict_12'
        }, inplace=True)
        submit_f.to_feather(SUBMIT_F)
        print(SUBMIT_F, submit_f.shape)

        # 球種予測で使用
        if not use_sub_model:
            train_predict = lgb_model.predict(train_d, num_iteration = lgb_model.best_iteration)

            df_train_predict = pd.DataFrame(train_predict).reset_index()
            submodel = pd.concat([df_train_predict, submit], ignore_index=True)
            submodel.drop(columns=['index'], inplace=True)
            submodel.rename(columns={
                0: 'predict_0', 1: 'predict_1', 2: 'predict_2', 3: 'predict_3',
                4: 'predict_4', 5: 'predict_5', 6: 'predict_6', 7: 'predict_7',
                8: 'predict_8', 9: 'predict_9', 10: 'predict_10', 11: 'predict_11', 12: 'predict_12'
            }, inplace=True)
            
            submodel.to_feather(OUT_SUBMODEL)
            print(OUT_SUBMODEL, submodel.shape)

    column_cnt = len(train_d.columns)

    # 結果まとめ
    result = common.SUBMIT_COURSE_F.format(model_No, model_No, 1)
    print(result)
    df = pd.read_feather(result)
    columns = ['predict_0', 'predict_1', 'predict_2', 'predict_3', 
               'predict_4', 'predict_5', 'predict_6', 'predict_7',
              'predict_8', 'predict_9', 'predict_10', 'predict_11', 'predict_12']

    for i in range(2, common.DIVIDE_NUM+1):
        result = common.SUBMIT_COURSE_F.format(model_No, model_No, i)
        print(result)
        temp = pd.read_feather(result)
        for c in columns:
            df[c] = df[c] + temp[c]

    for c in columns:
        df[c] = df[c] / common.DIVIDE_NUM
    
    cv_ave = 0
    for cv in best_cv:
        cv_ave = cv_ave + cv
    
    cv_ave = cv_ave / common.DIVIDE_NUM
    print('CV(ave) = {}'.format(cv_ave))

    # 出力
    df = df.reset_index()
    df.to_csv(SUBMIT, header=False, index=False)
    print(SUBMIT)

    print('signate submit --competition-id=276 ./{} --note feat={}_cv={}'.format(SUBMIT, column_cnt, cv_ave))