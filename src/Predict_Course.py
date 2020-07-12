# coding:utf-8
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import feather
import common
import time


def preprocess(model_No, sample_No, use_sub_model, use_RLHL=False):

    if use_sub_model:
        ALL_MERGE = common.ALL_MERGE_SUB.format(model_No, model_No, sample_No)
    else:
        ALL_MERGE = common.ALL_MERGE.format(model_No, model_No, sample_No)

    all_pitch = pd.read_feather(ALL_MERGE)

    # # sub-modelを使用するとき
    # if use_sub_model:
    #     all_pitch['predict_curve'] = all_pitch['predict_curve'] / all_pitch['predict_straight']
    #     all_pitch['predict_slider'] = all_pitch['predict_slider'] / all_pitch['predict_straight']
    #     all_pitch['predict_shoot'] = all_pitch['predict_shoot'] / all_pitch['predict_straight']
    #     all_pitch['predict_fork'] = all_pitch['predict_fork'] / all_pitch['predict_straight']
    #     all_pitch['predict_changeup'] = all_pitch['predict_changeup'] / all_pitch['predict_straight']
    #     all_pitch['predict_sinker'] = all_pitch['predict_sinker'] / all_pitch['predict_straight']
    #     all_pitch['predict_cutball'] = all_pitch['predict_cutball'] / all_pitch['predict_straight']
    #     all_pitch.drop(columns=['predict_straight'], inplace=True)

    if use_RLHL:
        cond1 = all_pitch.columns.str.contains('_str_pit')
        cond2 = all_pitch.columns.str.contains('_ball_pit')
        cond3 = all_pitch.columns.str.contains('_str_bat')
        cond4 = all_pitch.columns.str.contains('_ball_bat')
        cond = ~(cond1 | cond2 | cond3 | cond4)
        all_pitch = all_pitch.loc[:, cond]

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

    return train_d, test_d, train['course']


def tuning(model_No, use_sub_model, boosting, metric):
    
    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    sample_No = 1
    train_d, _, train_y = preprocess(model_No, sample_No, use_sub_model)

    filename = '../submit/{}/tuning_course_{}_{}.log'.format(model_No, boosting, metric)
    common.tuning(train_d, train_y, 13, boosting, metric, filename)


def train_predict(model_No, use_sub_model, boosting, metric):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    start = time.time()
    best_cv = []
    
    for sample_No in range(1, common.DIVIDE_NUM+1):

        if use_sub_model:
            SUBMIT = common.SUBMIT_COURSE_SUB_CSV.format(model_No, model_No)
            FI_RESULT = common.FI_COURSE_SUB_F.format(model_No, sample_No)
        else:
            SUBMIT = common.SUBMIT_COURSE_CSV.format(model_No, model_No)
            FI_RESULT = common.FI_COURSE_F.format(model_No, sample_No)
        
        SUBMIT_F = common.SUBMIT_COURSE_F.format(model_No, model_No, sample_No)
        # OUT_SUBMODEL = common.PREDICT_COURSE.format(model_No, model_No, sample_No)
        
        train_d, test_d, train_y = preprocess(model_No, sample_No, use_sub_model)

        lgb_param_gbdt = {
            'objective' : 'multiclass',
            'boosting_type': 'gbdt',
            'metric' : metric,
            'num_class' : 13,
            'seed' : 0,
            'learning_rate' : 0.1,
            'lambda_l1': 8.769293390201968, 
            'lambda_l2': 3.913949617576324e-05, 
            'num_leaves': 6, 
            'feature_fraction': 0.4, 
            'bagging_fraction': 0.8391111798378441, 
            'bagging_freq': 4, 
            'min_child_samples': 50
        }
        lgb_param_dart = {
            'objective' : 'multiclass',
            'boosting_type': 'dart',
            'metric' : metric,
            'num_class' : 13,
            'seed' : 0,
            'learning_rate' : 0.1,
            'lambda_l1': 8.074719414659954, 
            'lambda_l2': 1.5919119266007067, 
            'num_leaves': 6, 
            'feature_fraction': 0.516, 
            'bagging_fraction': 0.7965160701163017, 
            'bagging_freq': 5, 
            'min_child_samples': 20
        }
        is_cv = True
        if boosting == common.GBDT:
            lgb_param = lgb_param_gbdt
            iter_num = 10000
        else:
            lgb_param = lgb_param_dart
            if metric == common.M_LOGLOSS:
                iter_num = 2000
            else:
                iter_num = 1300
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
            0: 'predict_0', 1: 'predict_1', 2: 'predict_2', 3: 'predict_3',
            4: 'predict_4', 5: 'predict_5', 6: 'predict_6', 7: 'predict_7',
            8: 'predict_8', 9: 'predict_9', 10: 'predict_10', 11: 'predict_11', 12: 'predict_12'
        }, inplace=True)
        submit_f.to_feather(SUBMIT_F)
        print(SUBMIT_F, submit_f.shape)

        # 球種予測で使用
        # if not use_sub_model:
        #     train_predict = lgb_model.predict(train_d, num_iteration = lgb_model.best_iteration)

        #     df_train_predict = pd.DataFrame(train_predict).reset_index()
        #     submodel = pd.concat([df_train_predict, submit], ignore_index=True)
        #     submodel.drop(columns=['index'], inplace=True)
        #     submodel.rename(columns={
        #         0: 'predict_0', 1: 'predict_1', 2: 'predict_2', 3: 'predict_3',
        #         4: 'predict_4', 5: 'predict_5', 6: 'predict_6', 7: 'predict_7',
        #         8: 'predict_8', 9: 'predict_9', 10: 'predict_10', 11: 'predict_11', 12: 'predict_12'
        #     }, inplace=True)
            
        #     submodel.to_feather(OUT_SUBMODEL)
        #     print(OUT_SUBMODEL, submodel.shape)

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
    print('Predict_Course: {} [s]'.format(end - start))

    signate_command = 'signate submit --competition-id=276 ./{} --note {}_{}_feat={}_cv={}'.format(SUBMIT, boosting, metric, column_cnt, cv_ave)
    common.write_log(model_No, signate_command)


def train_predict2(model_No, use_sub_model, boosting, metric, LR_HL):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    start = time.time()
    best_cv = []
    common.write_log(model_No, 'train_predict2 {}, {}, {}'.format(boosting, metric, LR_HL))
    
    for sample_No in range(1, common.DIVIDE_NUM+1):

        FI_RESULT = '../submit/{}/course_fi_{}_{}.f'.format(model_No, LR_HL, sample_No)
        SUBMIT_F = '../submit/{}/course_{}_{}_{}.f'.format(model_No, model_No, LR_HL, sample_No)
        OUT_SUBMODEL = '../submit/{}/course_train_{}_{}_{}.f'.format(model_No, model_No, LR_HL, sample_No)
        
        train_d, test_d, train_y = preprocess(model_No, sample_No, use_sub_model, True)

        if LR_HL == 'LR':
            # 目的変数をコースの左右に変更
            # left_str: 0 <- 0, 1, 2
            # center_str: 1 <- 3, 4, 5
            # right_str: 2 <- 6, 7, 8
            # left_ball: 3 <- 9, 11
            # right_ball: 4 <- 10, 12
            convert = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 3, 4]
            rename_col = {0: 'left_str', 1: 'center_str', 2: 'right_str', 3: 'left_ball', 4: 'right_ball'}
        elif LR_HL == 'HL':
            # 目的変数をコースの上下に変更
            # high_str: 0 <- 0, 3, 6
            # mid_str: 1 <- 1, 4, 7
            # low_str: 2 <- 2, 5, 8
            # high_ball: 3 <- 9, 10
            # low_ball: 4 <- 11, 12
            convert = [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4]
            rename_col = {0: 'high_str', 1: 'mid_str', 2: 'low_str', 3: 'high_ball', 4: 'low_ball'}
        else:
            raise Exception('LR_HL error')

        train_y = train_y.map(lambda x: convert[int(x)])
        print(train_y.shape)

        lgb_param_gbdt = {
            'objective' : 'multiclass',
            'boosting_type': 'gbdt',
            'metric' : metric,
            'num_class' : 5,
            'seed' : 0,
            'learning_rate' : 0.1,
            'lambda_l1': 8.769293390201968, 
            'lambda_l2': 3.913949617576324e-05, 
            'num_leaves': 6, 
            'feature_fraction': 0.4, 
            'bagging_fraction': 0.8391111798378441, 
            'bagging_freq': 4, 
            'min_child_samples': 50
        }
        lgb_param_dart = {
            'objective' : 'multiclass',
            'boosting_type': 'dart',
            'metric' : metric,
            'num_class' : 5,
            'seed' : 0,
            'learning_rate' : 0.1,
            'lambda_l1': 8.074719414659954, 
            'lambda_l2': 1.5919119266007067, 
            'num_leaves': 6, 
            'feature_fraction': 0.516, 
            'bagging_fraction': 0.7965160701163017, 
            'bagging_freq': 5, 
            'min_child_samples': 20
        }
        is_cv = True
        if boosting == common.GBDT:
            lgb_param = lgb_param_gbdt
            iter_num = 10000
        else:
            lgb_param = lgb_param_dart
            if metric == common.M_LOGLOSS:
                iter_num = 2000
            else:
                iter_num = 1300
            if sample_No != 1:
                is_cv = False
        
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
        submit_f.rename(columns=rename_col, inplace=True)
        submit_f.to_feather(SUBMIT_F)
        print(SUBMIT_F, submit_f.shape)

        # train predict
        train_predict = lgb_model.predict(train_d, num_iteration = lgb_model.best_iteration)
        df_train_predict = pd.DataFrame(train_predict).reset_index()
        submodel = pd.concat([df_train_predict, submit], ignore_index=True)
        submodel.drop(columns=['index'], inplace=True)
        submodel.rename(columns=rename_col, inplace=True)
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
    print('Predict_Course: {} [s]'.format(end - start))


def ensemble_RLHL(model_No):
    COURSE_TRAIN = '../submit/{}/course_train_{}_{}_{}.f'

    best_cv = []

    for sample_No in range(1, common.DIVIDE_NUM+1):
        train1 = pd.read_feather(COURSE_TRAIN.format(model_No, model_No, 'LR', sample_No))
        print(train1.shape)
        train2 = pd.read_feather(COURSE_TRAIN.format(model_No, model_No, 'HL', sample_No))
        print(train2.shape)
        merge = train1.join(train2)
        print(merge.shape)

        col1 = train1.columns
        col2 = train2.columns
        for c1 in col1:
            for c2 in col2:
                merge[c1+'_x_'+c2] = merge[c1] * merge[c2]
        merge.drop(columns=col1, inplace=True)
        merge.drop(columns=col2, inplace=True)

        all_pitch = pd.read_feather(common.ALL_MERGE.format(model_No, model_No, sample_No))
        print(all_pitch.shape)

        train_test = all_pitch.query(common.divide_period_query_train(sample_No))
        print(train_test.shape)
        merge['course'] = train_test['course'].reset_index(drop=True)
        print(merge.shape)

        train = merge.dropna(subset=['course'])
        print(train.shape)
        test = merge[merge['course'].isnull()]
        print(test.shape)
        train_x = train.drop(['course'], axis=1)
        train_y = train['course']
        test_x = test.drop(['course'], axis=1)
        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)

        lgb_param = {
            'objective' : 'multiclass',
            'boosting_type': 'gbdt',
            'metric' : 'multi_logloss',
            'num_class' : 13,
            'seed' : 0,
            'learning_rate' : 0.01,
            'num_leaves': 32
        }

        lgb_train = lgb.Dataset(train_x, train_y)
        cv, best_iter = common.lightgbm_cv(lgb_param, lgb_train, 2000, 'multi_logloss')
        common.write_log(model_No, 'CV({}) = {}, best_iter = {}'.format(sample_No, cv, best_iter))
        best_cv.append(cv)

        lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=best_iter)
        predict = lgb_model.predict(test_x, num_iteration = lgb_model.best_iteration)

        submit = pd.DataFrame(predict)
        submit.reset_index(inplace=True)
        print(submit.shape)

        RLHL_F = common.SUBMIT_COURSE_RLHL_F.format(model_No, model_No, sample_No)
        submit_f = submit.drop(columns=['index'])
        submit_f.rename(columns={
            0: 'predict_0', 1: 'predict_1', 2: 'predict_2', 3: 'predict_3',
            4: 'predict_4', 5: 'predict_5', 6: 'predict_6', 7: 'predict_7',
            8: 'predict_8', 9: 'predict_9', 10: 'predict_10', 11: 'predict_11', 12: 'predict_12'
        }, inplace=True)
        submit_f.to_feather(RLHL_F)
        print(RLHL_F, submit_f.shape)

    # 結果まとめ
    # result = common.SUBMIT_COURSE_RLHL_F.format(model_No, model_No, 1)
    # print(result)
    # df = pd.read_feather(result)
    # columns = ['predict_0', 'predict_1', 'predict_2', 'predict_3', 
    #         'predict_4', 'predict_5', 'predict_6', 'predict_7',
    #         'predict_8', 'predict_9', 'predict_10', 'predict_11', 'predict_12']

    # for i in range(2, common.DIVIDE_NUM+1):
    #     result = common.SUBMIT_COURSE_RLHL_F.format(model_No, model_No, i)
    #     print(result)
    #     temp = pd.read_feather(result)
    #     for c in columns:
    #         df[c] = df[c] + temp[c]

    # for c in columns:
    #     df[c] = df[c] / common.DIVIDE_NUM

    # cv_ave = 0
    # for cv in best_cv:
    #     cv_ave = cv_ave + cv
    
    # if len(best_cv) > 0:
    #     cv_ave = cv_ave / len(best_cv)
    #     common.write_log(model_No, 'CV(ave) = {}'.format(cv_ave))
    
    # SUBMIT = '../submit/{}/course_RLHL_{}.csv'.format(model_No, model_No)
    # df = df.reset_index()
    # df.to_csv(SUBMIT, header=False, index=False)
    # print(SUBMIT)

    # common.write_log(model_No, 'signate submit --competition-id=276 ./{} --note RLHL_cv={}'.format(SUBMIT, cv_ave))