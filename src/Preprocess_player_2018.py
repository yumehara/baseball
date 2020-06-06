# coding:utf-8
import gc
import numpy as np
import pandas as pd
import feather
import common

def preprocess(model_No):
    
    test_pitch_org = pd.read_feather(common.TEST_PITCH)
    print(test_pitch_org.shape)

    predict_ball = pd.read_feather(common.BALL_2018.format(model_No))
    print(predict_ball.shape)

    predict_course = pd.read_feather(common.COURSE_2018.format(model_No))
    print(predict_course.shape)

    test_pitch_org = pd.concat([test_pitch_org, predict_ball], axis=1)
    test_pitch_org = pd.concat([test_pitch_org, predict_course], axis=1)
    print(test_pitch_org.shape)
    
    sample_No = 1

    for div in common.DIVIDE_TEST:

        # test_pitchを分割
        test_query = 'index <= {}'.format(div)
        test_pitch = test_pitch_org.query(test_query).copy()
        print(test_pitch.shape)

        # 出力先
        OUT_PIT = common.PLAYER_PIT_2018.format(sample_No)
        OUT_BAT = common.PLAYER_BAT_2018.format(sample_No)
        sample_No = sample_No + 1

        # 左右
        test_pitch.replace('左', 'L', inplace=True)
        test_pitch.replace('右', 'R', inplace=True)
        test_pitch['pit_bat'] = test_pitch['投手投球左右'] + '_' + test_pitch['打者打席左右']

        # 投手
        # 球種
        pitch_ball = test_pitch[['投手ID','pit_bat',
                                'predict_straight','predict_curve','predict_slider','predict_shoot',
                                'predict_fork','predict_changeup','predict_sinker','predict_cutball']]
        groupby_pit = pitch_ball.groupby(['投手ID','pit_bat']).sum()
        pitch_ball = groupby_pit.reset_index(inplace=False)
        pitch_ball['ball_total'] = (pitch_ball['predict_straight'] + pitch_ball['predict_curve'] + pitch_ball['predict_slider'] + pitch_ball['predict_shoot']
                                + pitch_ball['predict_fork'] + pitch_ball['predict_changeup'] + pitch_ball['predict_sinker'] + pitch_ball['predict_cutball'])

        # コース
        pitch_course = test_pitch[['投手ID','pit_bat',
                                    'predict_0','predict_1','predict_2','predict_3','predict_4','predict_5',
                                    'predict_6','predict_7','predict_8','predict_9','predict_10','predict_11','predict_12']]
        groupby_course = pitch_course.groupby(['投手ID','pit_bat']).sum()
        pitch_course = groupby_course.reset_index(inplace=False)

        # 登板試合数
        pit_game = test_pitch[['投手ID', '試合ID']].groupby(['投手ID', '試合ID']).count()
        pit_game = pd.DataFrame(pit_game.groupby(['投手ID']).size())
        pit_game.reset_index(inplace=True)
        pit_game.rename(columns={0: 'pit_game_cnt'}, inplace=True)
        
        # イニング数
        pit_inning = test_pitch[['投手ID', '試合ID', 'イニング']].groupby(['投手ID', '試合ID', 'イニング']).count()
        pit_inning = pd.DataFrame(pit_inning.groupby(['投手ID']).size())
        pit_inning.reset_index(inplace=True)
        pit_inning.rename(columns={0: 'pit_inning_cnt'}, inplace=True)
        
        # 対戦打者数
        pit_batcnt = test_pitch[['投手ID', 'pit_bat', '試合ID', 'イニング', 'イニング内打席数']].groupby(['投手ID', 'pit_bat', '試合ID', 'イニング', 'イニング内打席数']).count()
        pit_batcnt = pd.DataFrame(pit_batcnt.groupby(['投手ID','pit_bat']).size())
        pit_batcnt.reset_index(inplace=True)
        pit_batcnt.rename(columns={0: 'pit_batter_cnt'}, inplace=True)

        # 投手実績まとめ
        pitch_ball = pitch_ball.merge(pitch_course, on=['投手ID','pit_bat'], how='left')
        pitch_ball = pitch_ball.merge(pit_game, on='投手ID', how='left')
        pitch_ball = pitch_ball.merge(pit_inning, on='投手ID', how='left')
        pitch_ball = pitch_ball.merge(pit_batcnt, on=['投手ID','pit_bat'], how='left')

        pitch_ball.rename(columns={
            'predict_straight':'straight', 'predict_curve':'curve', 'predict_slider':'slider', 'predict_shoot':'shoot',
            'predict_fork':'fork','predict_changeup':'changeup','predict_sinker':'sinker', 'predict_cutball':'cutball',
            'ball_total' : 'total',
            'predict_0':'course_0', 'predict_1':'course_1', 'predict_2':'course_2',
            'predict_3':'course_3', 'predict_4':'course_4', 'predict_5':'course_5',
            'predict_6':'course_6', 'predict_7':'course_7', 'predict_8':'course_8',
            'predict_9':'course_9', 'predict_10':'course_10', 'predict_11':'course_11','predict_12':'course_12'
        }, inplace=True)

        # 投手出力
        pitch_ball.to_feather(OUT_PIT)
        print(OUT_PIT, pitch_ball.shape)
        del pitch_ball

        # 野手
        # 打席数
        bat_ball = test_pitch[['打者ID', '試合ID', 'イニング', 'イニング内打席数']].groupby(['打者ID', '試合ID', 'イニング', 'イニング内打席数']).count()
        bat_ball = pd.DataFrame(bat_ball.groupby(['打者ID']).size())
        bat_ball.reset_index(inplace=True)
        bat_ball.rename(columns={0: 'batter_cnt'}, inplace=True)

        # 試合数
        bat_game = test_pitch[['打者ID', '試合ID']].groupby(['打者ID', '試合ID']).count()
        bat_game = pd.DataFrame(bat_game.groupby(['打者ID']).size())
        bat_game.reset_index(inplace=True)
        bat_game.rename(columns={0: 'bat_game_cnt'}, inplace=True)

        # 打者成績まとめ
        bat_ball = bat_ball.merge(bat_game, on='打者ID', how='left')

        # 野手出力
        bat_ball.to_feather(OUT_BAT)
        print(OUT_BAT, bat_ball.shape)
        del bat_ball
        del test_pitch

        
