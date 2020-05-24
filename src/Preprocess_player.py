# coding:utf-8
import gc
import numpy as np
import pandas as pd
import feather
import common

def preprocess(is_fill):

    train_player = pd.read_feather(common.TRAIN_PLAYER)
    test_player = pd.read_feather(common.TEST_PLAYER)
    print(train_player.shape)
    print(test_player.shape)

    all_player = train_player.append(test_player, ignore_index=True)
    print(all_player.shape)

    # 外国人助っ人
    all_player['foreigner']=0
    all_player.loc[all_player['出身国']!='日本', 'foreigner'] = 1

    # 社会人出身
    all_player['company'] = 0
    all_player.loc[~all_player['社会人'].isnull(), 'company'] = 1
    # 大卒
    all_player['univ']=0
    all_player.loc[all_player['出身大学ID']!=0, 'univ'] = 1
    # 高卒
    all_player['highsch'] = 0
    all_player.loc[(all_player['company']==0)&(all_player['univ']==0)&(all_player['foreigner']==0) , 'highsch'] = 1

    # 年齢
    all_player['birth_day'] = pd.to_datetime(all_player['生年月日'])
    all_player['age'] = all_player['年度'] - all_player['birth_day'].dt.year
    # 現役年数
    all_player['play_year'] = all_player['年度'] - all_player['ドラフト年']
    all_player.loc[all_player['ドラフト年'].isnull(), 'play_year'] = 6

    # 球団ごとの年棒順位
    all_player['salary_rank'] = all_player.groupby(['年度', 'チームID'])['年俸'].rank(ascending=False)
    all_player['rank_year'] = all_player['salary_rank']/ all_player['play_year'] 
    all_player['rank_x_year'] = all_player['salary_rank'] * all_player['play_year'] 
    all_player['salary_year'] = all_player['年俸']/ all_player['play_year'] 
    all_player['salary_x_year'] = all_player['年俸'] * all_player['play_year'] 

    # 身長・体重
    all_player['bmi'] = all_player['体重']*10000/(all_player['身長']*all_player['身長'])

    # 不要な列を削除
    all_player.drop(
        columns=[
            'チームID', 'チーム名', '選手名', '背番号', '打', '生年月日', 
            '出身高校ID', '出身高校名', '出身大学ID', '出身大学名', '社会人', 
            'ドラフト年', 'ドラフト種別', 
            '出身国', '出身地', '血液型', 'birth_day',
        ], inplace=True)

    # rename
    all_player.rename(columns={
        '育成選手F': 'firm',
        '身長': 'height',
        '体重': 'weight',
        'ドラフト順位': 'draft_order',
        '年俸': 'salary',
    }, inplace=True)


    for sample_No in range(1, common.DIVIDE_NUM+1):

        OUT_PITCHER = common.ALLPITCHER.format(sample_No)
        OUT_ALLPLAYER = common.ALLPLAYER.format(sample_No)

        # 2017年の成績(1/4ずつ)
        pit_2017 = pd.read_feather(common.PLAYER_PIT_2017.format(sample_No))
        bat_2017 = pd.read_feather(common.PLAYER_BAT_2017.format(sample_No))
        print(pit_2017.shape)
        print(bat_2017.shape)

        # 投手のみ
        all_pitcher = all_player[all_player['位置']=='投手'].copy()

        dummy = pd.DataFrame({
            '投': ['右', '右', '左', '左'],
            'pit_bat': ['R_L', 'R_R', 'L_L', 'L_R']
        })
        all_pitcher = all_pitcher.merge(dummy, on='投', how='outer')

        all_pitcher = all_pitcher.merge(pit_2017, left_on=['選手ID','pit_bat'], right_on=['投手ID','pit_bat'], how='left')
        all_pitcher.loc[(all_pitcher['投手ID'].isnull()) & (all_pitcher['foreigner']==1), '投手ID'] = -1
        all_pitcher.loc[all_pitcher['投手ID'].isnull(), '投手ID'] = 0

        # 情報がない投手
        if is_fill:
            RightLeft = ['R_L', 'R_R', 'L_R', 'L_L']
            # 日本人平均
            for RL in RightLeft:
                pit_mean = all_pitcher[(all_pitcher['foreigner']==0)&(all_pitcher['投手ID']!=0)&(all_pitcher['pit_bat']==RL)].mean()
                condition = (all_pitcher['投手ID']==0)&(all_pitcher['pit_bat']==RL)
                # 平均で埋める
                fill_ball(condition, pit_mean, all_pitcher)

            #外国人平均
            for RL in RightLeft:
                pit_mean = all_pitcher[(all_pitcher['foreigner']==1)&(all_pitcher['投手ID']!=-1)&(all_pitcher['pit_bat']==RL)].mean()
                condition = (all_pitcher['投手ID']==-1)&(all_pitcher['pit_bat']==RL)
                # 平均で埋める
                fill_ball(condition, pit_mean, all_pitcher)

        # 各球種のストレートに対する比率
        ball_not_straight = ['curve', 'slider', 'shoot', 'fork', 'changeup', 'sinker', 'cutball']
        for ball in ball_not_straight:
            all_pitcher[ball] = all_pitcher[ball] / all_pitcher['straight']

        # コースの比率
        course_kind = ['course_0', 'course_1', 'course_2', 'course_3', 'course_4', 'course_5', 'course_6', 
                        'course_7', 'course_8', 'course_9', 'course_10', 'course_11', 'course_12']
        for course in course_kind:
            all_pitcher[course] = all_pitcher[course] / all_pitcher['total']

        # コースの種類
        all_pitcher['high_str'] = all_pitcher['course_0'] + all_pitcher['course_3'] + all_pitcher['course_6'] 
        all_pitcher['high_ball'] = all_pitcher['course_9'] + all_pitcher['course_10'] 
        all_pitcher['mid_str'] = all_pitcher['course_1'] + all_pitcher['course_4'] + all_pitcher['course_7'] 
        all_pitcher['low_str'] = all_pitcher['course_2'] + all_pitcher['course_5'] + all_pitcher['course_8'] 
        all_pitcher['low_ball'] = all_pitcher['course_11'] + all_pitcher['course_12'] 

        all_pitcher['left_str'] = all_pitcher['course_0'] + all_pitcher['course_1'] + all_pitcher['course_2'] 
        all_pitcher['left_ball'] = all_pitcher['course_9'] + all_pitcher['course_11'] 
        all_pitcher['center_str'] = all_pitcher['course_3'] + all_pitcher['course_4'] + all_pitcher['course_5'] 
        all_pitcher['right_str'] = all_pitcher['course_6'] + all_pitcher['course_7'] + all_pitcher['course_8'] 
        all_pitcher['right_ball'] = all_pitcher['course_10'] + all_pitcher['course_12'] 

        # 不要な列を削除
        all_pitcher.drop(columns=[
            'straight', '投手ID', '位置', '投',
            'course_0', 'course_1', 'course_2', 'course_3', 'course_4', 'course_5', 'course_6', 
            'course_7', 'course_8', 'course_9', 'course_10', 'course_11', 'course_12'
        ], inplace=True)

        # 投手のみ出力
        all_pitcher.to_feather(OUT_PITCHER)
        print(OUT_PITCHER, all_pitcher.shape)
        del all_pitcher


        # 打者(全選手)
        all_batter = all_player.copy()
        all_batter = all_batter.merge(bat_2017, left_on='選手ID', right_on='打者ID', how='left')

        all_batter.loc[all_batter['打者ID'].isnull(), '打者ID'] = 0

        # 情報がない打者
        if is_fill:
            # 投手以外
            bat_mean = all_batter[(all_batter['打者ID']!=0)&(all_batter['位置']!='投手')].mean()
            condition = (all_batter['打者ID']==0)&(all_batter['位置']!='投手')
            # 平均で埋める
            all_batter.loc[condition, 'batter_cnt'] = bat_mean['batter_cnt']
            all_batter.loc[condition, 'bat_game_cnt'] = bat_mean['bat_game_cnt']

            # 投手
            bat_mean = all_batter[(all_batter['打者ID']!=0)&(all_batter['位置']=='投手')].mean()
            condition = (all_batter['打者ID']==0)&(all_batter['位置']=='投手')
            # 平均で埋める
            all_batter.loc[condition, 'batter_cnt'] = bat_mean['batter_cnt']
            all_batter.loc[condition, 'bat_game_cnt'] = bat_mean['bat_game_cnt']

        # 不要な列を削除
        all_batter.drop(columns=['打者ID', '位置', '投'], inplace=True)

        # 全選手出力
        all_batter.to_feather(OUT_ALLPLAYER)
        print(OUT_ALLPLAYER, all_batter.shape)
        del all_batter

    del all_player


# 情報がない選手
def fill_ball(condition, source, target):
    ball_kind = [
        'straight', 'curve', 'slider', 'shoot', 'fork', 'changeup', 'sinker', 'cutball', 'total',
        'pit_game_cnt', 'pit_inning_cnt', 'pit_batter_cnt',
        'course_0', 'course_1', 'course_2', 'course_3', 'course_4', 'course_5', 'course_6', 
        'course_7', 'course_8', 'course_9', 'course_10', 'course_11', 'course_12'
    ]
    for ball in ball_kind:
        target.loc[condition, ball] = source[ball]