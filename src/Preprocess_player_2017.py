# coding:utf-8
import gc
import numpy as np
import pandas as pd
import feather
import common

def preprocess():
    
    train_pitch_org = pd.read_feather(common.TRAIN_PITCH)
    print(train_pitch_org.shape)

    for sample_No in range(1, common.DIVIDE_NUM+1):

        # train_pitchを分割
        train_pitch = train_pitch_org.query(common.divide_period_query_pre(sample_No)).copy()

        # 出力先
        OUT_PIT = common.PLAYER_PIT_2017.format(sample_No)
        OUT_CAT = common.PLAYER_CAT_2017.format(sample_No)
        OUT_BAT = common.PLAYER_BAT_2017.format(sample_No)

        # 左右
        train_pitch.replace('左', 'L', inplace=True)
        train_pitch.replace('右', 'R', inplace=True)
        train_pitch['pit_bat'] = train_pitch['投手投球左右'] + '_' + train_pitch['打者打席左右']

        train_pitch.rename(columns={'球種': 'ball', '投球位置区域': 'course'}, inplace=True)

        # 投手
        # 球種
        pitch_ball = train_pitch[['投手ID','pit_bat','ball']]
        pitch_ball = pd.get_dummies(pitch_ball, columns=['ball'])

        groupby_pit = pitch_ball.groupby(['投手ID','pit_bat']).sum()
        groupby_pit.rename(columns={
            'ball_0': 'straight',
            'ball_1': 'curve',
            'ball_2': 'slider',
            'ball_3': 'shoot',
            'ball_4': 'fork',
            'ball_5': 'changeup',
            'ball_6': 'sinker',
            'ball_7': 'cutball',
        }, inplace=True)
        pitch_ball = groupby_pit.reset_index(inplace=False)

        pitch_ball['total'] = (pitch_ball['straight'] + pitch_ball['curve'] + pitch_ball['slider'] + pitch_ball['shoot'] 
                            + pitch_ball['fork'] + pitch_ball['changeup'] + pitch_ball['sinker'] + pitch_ball['cutball'])

        # コース
        pitch_course = train_pitch[['投手ID','pit_bat','course']]
        pitch_course = pd.get_dummies(pitch_course, columns=['course'])
        groupby_course = pitch_course.groupby(['投手ID','pit_bat']).sum()
        pitch_course = groupby_course.reset_index(inplace=False)

        # 登板試合数
        pit_game = train_pitch[['投手ID', '試合ID']].groupby(['投手ID', '試合ID']).count()
        pit_game = pd.DataFrame(pit_game.groupby(['投手ID']).size())
        pit_game.reset_index(inplace=True)
        pit_game.rename(columns={0: 'pit_game_cnt'}, inplace=True)

        # イニング数
        pit_inning = train_pitch[['投手ID', '試合ID', 'イニング']].groupby(['投手ID', '試合ID', 'イニング']).count()
        pit_inning = pd.DataFrame(pit_inning.groupby(['投手ID']).size())
        pit_inning.reset_index(inplace=True)
        pit_inning.rename(columns={0: 'pit_inning_cnt'}, inplace=True)

        # 対戦打者数
        pit_batcnt = train_pitch[['投手ID', 'pit_bat', '試合ID', 'イニング', 'イニング内打席数']].groupby(['投手ID', 'pit_bat', '試合ID', 'イニング', 'イニング内打席数']).count()
        pit_batcnt = pd.DataFrame(pit_batcnt.groupby(['投手ID','pit_bat']).size())
        pit_batcnt.reset_index(inplace=True)
        pit_batcnt.rename(columns={0: 'pit_batter_cnt'}, inplace=True)

        # 投手実績まとめ
        pitch_ball = pitch_ball.merge(pitch_course, on=['投手ID','pit_bat'], how='left')
        pitch_ball = pitch_ball.merge(pit_game, on='投手ID', how='left')
        pitch_ball = pitch_ball.merge(pit_inning, on='投手ID', how='left')
        pitch_ball = pitch_ball.merge(pit_batcnt, on=['投手ID','pit_bat'], how='left')

        # 投手出力
        pitch_ball.to_feather(OUT_PIT)
        print(OUT_PIT, pitch_ball.shape)
        del pitch_ball

        # 捕手
        # 球種
        catch_ball = train_pitch[['捕手ID','pit_bat','ball']]
        catch_ball = pd.get_dummies(catch_ball, columns=['ball'])

        groupby_cat = catch_ball.groupby(['捕手ID','pit_bat']).sum()
        groupby_cat.rename(columns={
            'ball_0': 'straight',
            'ball_1': 'curve',
            'ball_2': 'slider',
            'ball_3': 'shoot',
            'ball_4': 'fork',
            'ball_5': 'changeup',
            'ball_6': 'sinker',
            'ball_7': 'cutball',
        }, inplace=True)
        catch_ball = groupby_cat.reset_index(inplace=False)

        catch_ball['total'] = (catch_ball['straight'] + catch_ball['curve'] + catch_ball['slider'] + catch_ball['shoot'] 
                            + catch_ball['fork'] + catch_ball['changeup'] + catch_ball['sinker'] + catch_ball['cutball'])

        # コース
        catch_course = train_pitch[['捕手ID','pit_bat','course']]
        catch_course = pd.get_dummies(catch_course, columns=['course'])
        catch_course = catch_course.groupby(['捕手ID','pit_bat']).sum().reset_index()

        # 捕手実績まとめ
        catch_ball = catch_ball.merge(catch_course, on=['捕手ID','pit_bat'], how='left')

        # 捕手出力
        catch_ball.to_feather(OUT_CAT)
        print(OUT_CAT, catch_ball.shape)
        del catch_ball

        # 野手
        # 打席数
        bat_ball = train_pitch[['打者ID', '試合ID', 'イニング', 'イニング内打席数']].groupby(['打者ID', '試合ID', 'イニング', 'イニング内打席数']).count()
        bat_ball = pd.DataFrame(bat_ball.groupby(['打者ID']).size())
        bat_ball.reset_index(inplace=True)
        bat_ball.rename(columns={0: 'batter_cnt'}, inplace=True)

        # 試合数
        bat_game = train_pitch[['打者ID', '試合ID']].groupby(['打者ID', '試合ID']).count()
        bat_game = pd.DataFrame(bat_game.groupby(['打者ID']).size())
        bat_game.reset_index(inplace=True)
        bat_game.rename(columns={0: 'bat_game_cnt'}, inplace=True)

        # 打者成績まとめ
        bat_ball = bat_ball.merge(bat_game, on='打者ID', how='left')

        # 野手出力
        bat_ball.to_feather(OUT_BAT)
        print(OUT_BAT, bat_ball.shape)
        del bat_ball
        del train_pitch