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
        pitch_ball = ball_grouping(train_pitch, '投手ID')

        # コース
        pitch_course = course_grouping(train_pitch, '投手ID')

        # 登板試合数
        pit_game = game_count(train_pitch, '投手ID', 'pit_game_cnt')

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
        catch_ball = ball_grouping(train_pitch, '捕手ID')

        # コース
        catch_course = course_grouping(train_pitch, '捕手ID')

        # 出場試合数
        catch_game = game_count(train_pitch, '捕手ID', 'cat_game_cnt')

        # 捕手実績まとめ
        catch_ball = catch_ball.merge(catch_course, on=['捕手ID','pit_bat'], how='left')
        catch_ball = catch_ball.merge(catch_game, on='捕手ID', how='left')

        # 捕手出力
        catch_ball.to_feather(OUT_CAT)
        print(OUT_CAT, catch_ball.shape)
        del catch_ball

        # 野手
        # 球種
        batter_ball = ball_grouping(train_pitch, '打者ID')

        # コース
        batter_course = course_grouping(train_pitch, '打者ID')

        # 打席数
        bat_ball = train_pitch[['打者ID', '試合ID', 'イニング', 'イニング内打席数']].groupby(['打者ID', '試合ID', 'イニング', 'イニング内打席数']).count()
        bat_ball = pd.DataFrame(bat_ball.groupby(['打者ID']).size())
        bat_ball.reset_index(inplace=True)
        bat_ball.rename(columns={0: 'batter_cnt'}, inplace=True)

        # 出場試合数
        bat_game = game_count(train_pitch, '打者ID', 'bat_game_cnt')

        # 打者成績まとめ
        batter_ball = batter_ball.merge(batter_course, on=['打者ID','pit_bat'], how='left')
        batter_ball = batter_ball.merge(bat_ball, on='打者ID', how='left')
        batter_ball = batter_ball.merge(bat_game, on='打者ID', how='left')

        # 野手出力
        batter_ball.to_feather(OUT_BAT)
        print(OUT_BAT, batter_ball.shape)
        del batter_ball
        del train_pitch

def ball_grouping(source, group_col):
    pitch_ball = source[[group_col,'pit_bat','ball']]
    pitch_ball = pd.get_dummies(pitch_ball, columns=['ball'])

    groupby_pit = pitch_ball.groupby([group_col,'pit_bat']).sum()
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
    return pitch_ball

def course_grouping(source, group_col):
    pitch_course = source[[group_col,'pit_bat','course']]
    pitch_course = pd.get_dummies(pitch_course, columns=['course'])
    pitch_course = pitch_course.groupby([group_col,'pit_bat']).sum().reset_index()
    return pitch_course

def game_count(source, group_col, target_col):
    pit_game = source[[group_col, '試合ID']].groupby([group_col, '試合ID']).count()
    pit_game = pd.DataFrame(pit_game.groupby([group_col]).size())
    pit_game.reset_index(inplace=True)
    pit_game.rename(columns={0: target_col}, inplace=True)
    return pit_game
