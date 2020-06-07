# coding:utf-8
import gc
import numpy as np
import pandas as pd
import feather
import common

def preprocess(model_No):

    test_pitch_org = pd.read_feather(common.TEST_PITCH)
    print(test_pitch_org.shape)

    predict_ball = pd.read_feather(common.PREDICT_BALL_2018.format(model_No))
    print(predict_ball.shape)

    predict_course = pd.read_feather(common.PREDICT_COURSE_2018.format(model_No))
    print(predict_course.shape)

    test_pitch_org = pd.concat([test_pitch_org, predict_ball], axis=1)
    test_pitch_org = pd.concat([test_pitch_org, predict_course], axis=1)
    print(test_pitch_org.shape)
    
    sample_No = 1

    for div in common.DIVIDE_TEST:

        OUTPUT = common.BALL_2018.format(sample_No)
        sample_No = sample_No + 1

        # test_pitchを分割
        test_query = 'index <= {}'.format(div)
        test_pitch = test_pitch_org.query(test_query).copy()
        print(test_pitch.shape)

        test_pitch['ball_cnt'] = test_pitch['プレイ前ストライク数'].astype(str) + '-' + test_pitch['プレイ前ボール数'].astype(str)

        test_pitch.replace('左', 'L', inplace=True)
        test_pitch.replace('右', 'R', inplace=True)
        test_pitch['pit_bat'] = test_pitch['投手投球左右'] + '_' + test_pitch['打者打席左右']

        # 球種
        pitch_ball = test_pitch[['ball_cnt','pit_bat',
                                'predict_straight','predict_curve','predict_slider','predict_shoot',
                                'predict_fork','predict_changeup','predict_sinker','predict_cutball']]
        groupby_pit = pitch_ball.groupby(['ball_cnt', 'pit_bat']).sum()
        pitch_ball = groupby_pit.reset_index()
        pitch_ball['ball_total'] = (pitch_ball['predict_straight'] + pitch_ball['predict_curve'] + pitch_ball['predict_slider'] + pitch_ball['predict_shoot']
                                    + pitch_ball['predict_fork'] + pitch_ball['predict_changeup'] + pitch_ball['predict_sinker'] + pitch_ball['predict_cutball'])

        pitch_ball['bc_straight'] = pitch_ball['predict_straight']/pitch_ball['ball_total']
        pitch_ball['bc_curve'] = pitch_ball['predict_curve']/pitch_ball['ball_total']
        pitch_ball['bc_slider'] = pitch_ball['predict_slider']/pitch_ball['ball_total']
        pitch_ball['bc_shoot'] = pitch_ball['predict_shoot']/pitch_ball['ball_total']
        pitch_ball['bc_fork'] = pitch_ball['predict_fork']/pitch_ball['ball_total']
        pitch_ball['bc_changeup'] = pitch_ball['predict_changeup']/pitch_ball['ball_total']
        pitch_ball['bc_sinker'] = pitch_ball['predict_sinker']/pitch_ball['ball_total']
        pitch_ball['bc_cutball'] = pitch_ball['predict_cutball']/pitch_ball['ball_total']

        pitch_ball.drop(columns=[
            'ball_total',
            'predict_straight', 'predict_curve', 'predict_slider', 'predict_shoot',
            'predict_fork', 'predict_changeup', 'predict_sinker', 'predict_cutball'
        ], inplace=True)

        print(pitch_ball.shape)

        # コース
        pitch_course = test_pitch[['ball_cnt','pit_bat',
                            'predict_0','predict_1','predict_2','predict_3','predict_4','predict_5',
                            'predict_6','predict_7','predict_8','predict_9','predict_10','predict_11','predict_12']]
        groupby_course = pitch_course.groupby(['ball_cnt','pit_bat']).sum()
        pitch_course = groupby_course.reset_index(inplace=False)
        pitch_course['total'] = (pitch_course['predict_0'] + pitch_course['predict_1'] + pitch_course['predict_2'] + pitch_course['predict_3'] + 
        pitch_course['predict_4'] + pitch_course['predict_5'] + pitch_course['predict_6'] + pitch_course['predict_7'] + 
        pitch_course['predict_8'] + pitch_course['predict_9'] + pitch_course['predict_10'] + pitch_course['predict_11'] + pitch_course['predict_12'])

        pitch_course['bc_course_0'] = pitch_course['predict_0']/pitch_course['total']
        pitch_course['bc_course_1'] = pitch_course['predict_1']/pitch_course['total']
        pitch_course['bc_course_2'] = pitch_course['predict_2']/pitch_course['total']
        pitch_course['bc_course_3'] = pitch_course['predict_3']/pitch_course['total']
        pitch_course['bc_course_4'] = pitch_course['predict_4']/pitch_course['total']
        pitch_course['bc_course_5'] = pitch_course['predict_5']/pitch_course['total']
        pitch_course['bc_course_6'] = pitch_course['predict_6']/pitch_course['total']
        pitch_course['bc_course_7'] = pitch_course['predict_7']/pitch_course['total']
        pitch_course['bc_course_8'] = pitch_course['predict_8']/pitch_course['total']
        pitch_course['bc_course_9'] = pitch_course['predict_9']/pitch_course['total']
        pitch_course['bc_course_10'] = pitch_course['predict_10']/pitch_course['total']
        pitch_course['bc_course_11'] = pitch_course['predict_11']/pitch_course['total']
        pitch_course['bc_course_12'] = pitch_course['predict_12']/pitch_course['total']

        pitch_course.drop(columns=[
            'predict_0', 'predict_1', 'predict_2', 'predict_3', 'predict_4', 'predict_5', 
            'predict_6', 'predict_7', 'predict_8', 'predict_9', 'predict_10', 'predict_11', 'predict_12'], inplace=True)

        # コースの種類
        pitch_course['bc_high_str'] = pitch_course['bc_course_0'] + pitch_course['bc_course_3'] + pitch_course['bc_course_6'] 
        pitch_course['bc_high_ball'] = pitch_course['bc_course_9'] + pitch_course['bc_course_10'] 
        pitch_course['bc_mid_str'] = pitch_course['bc_course_1'] + pitch_course['bc_course_4'] + pitch_course['bc_course_7'] 
        pitch_course['bc_low_str'] = pitch_course['bc_course_2'] + pitch_course['bc_course_5'] + pitch_course['bc_course_8'] 
        pitch_course['bc_low_ball'] = pitch_course['bc_course_11'] + pitch_course['bc_course_12'] 

        pitch_course['bc_left_str'] = pitch_course['bc_course_0'] + pitch_course['bc_course_1'] + pitch_course['bc_course_2'] 
        pitch_course['bc_left_ball'] = pitch_course['bc_course_9'] + pitch_course['bc_course_11'] 
        pitch_course['bc_center_str'] = pitch_course['bc_course_3'] + pitch_course['bc_course_4'] + pitch_course['bc_course_5'] 
        pitch_course['bc_right_str'] = pitch_course['bc_course_6'] + pitch_course['bc_course_7'] + pitch_course['bc_course_8'] 
        pitch_course['bc_right_ball'] = pitch_course['bc_course_10'] + pitch_course['bc_course_12'] 

        # pitch_course.drop(columns=[
        #     'bc_course_0', 'bc_course_1', 'bc_course_2', 'bc_course_3', 'bc_course_4', 'bc_course_5', 
        #     'bc_course_6', 'bc_course_7', 'bc_course_8', 'bc_course_9', 'bc_course_10', 'bc_course_11', 'bc_course_12'], inplace=True)

        print(pitch_course.shape)

        # マージ
        pitch_ball = pitch_ball.merge(pitch_course, on=['ball_cnt', 'pit_bat'], how='left')
        print(pitch_ball.shape)

        # 出力
        pitch_ball.to_feather(OUTPUT)
