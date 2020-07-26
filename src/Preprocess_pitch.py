# coding:utf-8
import gc
import numpy as np
import pandas as pd
import feather
import common

def preprocess():
    
    train_pitch = pd.read_feather(common.TRAIN_PITCH)
    test_pitch = pd.read_feather(common.TEST_PITCH)

    INPUT_BALL2017 = common.BALL_2017
    OUTPUT = common.ALL_PITCH

    test_pitch['球種'] = None
    test_pitch['投球位置区域'] = None
    print(train_pitch.shape)
    print(test_pitch.shape)

    all_pitch = train_pitch.append(test_pitch, ignore_index=True)
    print(all_pitch.shape)
    # 球種
    all_pitch.rename(columns={'球種': 'ball', '投球位置区域': 'course'}, inplace=True)
    # ボールカウント
    all_pitch['ball_cnt'] = all_pitch['プレイ前ストライク数'].astype(str) + '-' + all_pitch['プレイ前ボール数'].astype(str)
    # 左右
    all_pitch.replace('左', 'L', inplace=True)
    all_pitch.replace('右', 'R', inplace=True)
    all_pitch['pit_bat'] = all_pitch['投手投球左右'] + '_' + all_pitch['打者打席左右']
    # all_pitch.loc[all_pitch['投手投球左右']=='L', 'pitch_LR']=1
    # all_pitch.loc[all_pitch['投手投球左右']=='R', 'pitch_LR']=0
    # all_pitch.loc[all_pitch['打者打席左右']=='L', 'bat_LR']=1
    # all_pitch.loc[all_pitch['打者打席左右']=='R', 'bat_LR']=0
    # 2017年のデータをマージ
    train_ball = pd.read_feather(INPUT_BALL2017)
    all_pitch = all_pitch.merge(train_ball, on=['ball_cnt', 'pit_bat'], how='left')
    all_pitch = all_pitch.drop(columns=['ball_cnt'])
    # 一塁走者ID, 二塁走者ID, 三塁走者ID
    all_pitch['first'] = 0
    all_pitch['second'] = 0
    all_pitch['third'] = 0
    all_pitch.loc[~np.isnan(all_pitch['一塁走者ID']), 'first'] = 1
    all_pitch.loc[~np.isnan(all_pitch['二塁走者ID']), 'second'] = 1
    all_pitch.loc[~np.isnan(all_pitch['三塁走者ID']), 'third'] = 1
    all_pitch['base_cnt'] = all_pitch['first'] + all_pitch['second'] + all_pitch['third']
    # 表裏
    all_pitch['top_bot']=0
    all_pitch.loc[all_pitch['表裏']=='裏', 'top_bot']=1
    # 投手役割
    # all_pitch['role'] = 0
    # all_pitch.loc[all_pitch['投手役割']=='先発', 'role']=1
    # 打者守備位置
    # all_pitch['pos_pit']=0
    # all_pitch.loc[all_pitch['打者守備位置']=='投手', 'pos_pit']=1
    # 開幕からの日数
    date_min = all_pitch.groupby('年度').agg({'日付': min})
    date_min.rename(columns={'日付': 'opening_date'}, inplace=True)
    date_min.reset_index(inplace=True)
    date_min['opening_date'] = pd.to_datetime(date_min['opening_date'])
    all_pitch = pd.merge(all_pitch, date_min, on='年度', how='left')
    all_pitch['game_date'] = pd.to_datetime(all_pitch['日付'])
    all_pitch['date_from_opening'] = (all_pitch['game_date'] - all_pitch['opening_date']).dt.days
    # 試合開始からの経過時間
    time_min = all_pitch.groupby('試合ID').agg({'時刻': min})
    time_min.rename(columns={'時刻': 'start_time'}, inplace=True)
    time_min.reset_index(inplace=True)
    time_min['start_time'] = pd.to_datetime(time_min['start_time'])
    all_pitch = pd.merge(all_pitch, time_min, on='試合ID', how='left')
    all_pitch['game_time'] = pd.to_datetime(all_pitch['時刻'])
    all_pitch['elapsed_time'] = (all_pitch['game_time'] - all_pitch['start_time'])
    all_pitch['elapsed_min'] = all_pitch['elapsed_time'].dt.seconds / 60
    # 前の投球からの時間差
    min_diff = all_pitch.groupby(['試合ID'])['elapsed_min'].diff()
    all_pitch['min_diff'] = min_diff
    all_pitch.loc[all_pitch['投手イニング内投球数']==1, 'min_diff'] = np.NaN
    # イニングの通し番号
    all_pitch['total_inning'] = (all_pitch['イニング'] - 1) * 2 + all_pitch['top_bot']
    # イニング最初からの時間
    min_inning = all_pitch.groupby(['試合ID', 'total_inning']).agg({'elapsed_min': min, '試合内投球数': min})
    min_inning.reset_index(inplace=True)
    min_inning.rename(columns={'elapsed_min': 'start_inning', '試合内投球数': 'start_ball_inning'}, inplace=True)
    all_pitch = pd.merge(all_pitch, min_inning, on=['試合ID', 'total_inning'], how='left')
    all_pitch['elapsed_from_inning'] = (all_pitch['elapsed_min'] - all_pitch['start_inning'])
    all_pitch['ballnum_from_inning'] = (all_pitch['試合内投球数'] - all_pitch['start_ball_inning']) + 1
    all_pitch.drop(columns=['start_inning', 'start_ball_inning'], inplace=True)
    # 打席最初からの時間
    min_batter = all_pitch.groupby(['試合ID', 'total_inning', 'イニング内打席数']).agg({'elapsed_min': min})
    min_batter.rename(columns={'elapsed_min': 'start_batter'}, inplace=True)
    min_batter.reset_index(inplace=True)
    all_pitch = pd.merge(all_pitch, min_batter, on=['試合ID', 'total_inning', 'イニング内打席数'], how='left')
    all_pitch['elapsed_batter'] = (all_pitch['elapsed_min'] - all_pitch['start_batter'])
    all_pitch.drop(columns=['start_batter'], inplace=True)
    # 平均投球間隔
    all_pitch['ave_elapsed_game'] = (all_pitch['elapsed_min'] / (all_pitch['試合内投球数'] - 1))
    all_pitch['ave_elapsed_inning'] = (all_pitch['elapsed_from_inning'] / (all_pitch['ballnum_from_inning'] - 1))
    all_pitch['ave_elapsed_batter'] = (all_pitch['elapsed_batter'] / (all_pitch['打席内投球数'] - 1))
    # 平均投球間隔の差
    all_pitch['diff_elapsed_batter'] = all_pitch['min_diff'] - all_pitch['ave_elapsed_batter']
    all_pitch['diff_elapsed_inning'] = all_pitch['min_diff'] - all_pitch['ave_elapsed_inning']
    all_pitch['diff_elapsed_game'] = all_pitch['min_diff'] - all_pitch['ave_elapsed_game']
    # サヨナラの危機
    # all_pitch['sayonara'] = 0
    # all_pitch.loc[(all_pitch['イニング']>=9)&(all_pitch['表裏']=='裏'), 'sayonara']=1
    # 延長戦
    # all_pitch['extention'] = 0
    # all_pitch.loc[(all_pitch['イニング']>9), 'extention']=1
    # ナイター
    all_pitch['nighter'] = 0
    all_pitch.loc[all_pitch['game_time'].dt.hour>=18, 'nighter']=1
    # 交流戦
    # all_pitch['ce-pa'] = 0
    # all_pitch.loc[all_pitch['試合種別詳細']=='セ・パ交流戦', 'ce-pa']=1
    # リーグ
    # all_pitch['league'] = 0
    # all_pitch.loc[all_pitch['試合種別詳細']=='セ・リーグ公式戦', 'league']=1
    # ホーム・アウェー
    all_pitch['home']=0
    all_pitch.loc[all_pitch['投手チームID']==all_pitch['ホームチームID'], 'home'] = 1
    # 得点差
    point_diff = all_pitch['プレイ前ホームチーム得点数'] - all_pitch['プレイ前アウェイチーム得点数']
    all_pitch['point_diff'] = point_diff
    all_pitch.loc[all_pitch['home']==0, 'point_diff'] = -point_diff
    # 得点圏にランナーがいる
    all_pitch['runner_23'] = 0
    all_pitch.loc[(all_pitch['second']==1)|(all_pitch['third']==1), 'runner_23']=1
    # 送りバントの場面
    all_pitch['bant'] = 0
    all_pitch.loc[(all_pitch['first']==1)&(all_pitch['third']==0)&(all_pitch['プレイ前アウト数']==0)&(all_pitch['プレイ前ストライク数']<2), 'bant']=1
    # スクイズの場面
    # all_pitch['squize'] = 0
    # all_pitch.loc[(all_pitch['third']==1)&(all_pitch['プレイ前アウト数']<2)&(all_pitch['プレイ前ストライク数']<2), 'squize']=1
    # 上位打線
    # all_pitch['cleanup'] = 0
    # all_pitch.loc[(all_pitch['打者打順']>=1)&(all_pitch['打者打順']<=5), 'cleanup']=1
    # 失点ピンチ
    # all_pitch['pinch'] = 0
    # all_pitch.loc[(all_pitch['runner_23']==1)&(all_pitch['cleanup']==1), 'pinch']=1
    # 押出しの危機
    # all_pitch['fourball'] = 0
    # all_pitch.loc[(all_pitch['base_cnt']==3)&(all_pitch['プレイ前ボール数']>1), 'fourball']=1
    # セーブがつく場面
    # all_pitch['savepoint'] = 0
    # all_pitch.loc[(all_pitch['イニング']>=9)&(all_pitch['point_diff']<4), 'savepoint']=1
    # 1つ前の投球・ファウル数
    all_pitch['ball_count_sum'] =  all_pitch['プレイ前ボール数'] + all_pitch['プレイ前ストライク数']
    groupby_batter = all_pitch.groupby(['試合ID', 'イニング', 'イニング内打席数'])
    all_pitch['pre_ball_foul'] = 1 - groupby_batter['ball_count_sum'].diff().fillna(1)
    all_pitch['pre_ball_ball'] = groupby_batter['プレイ前ボール数'].diff().fillna(0)
    all_pitch['pre_ball_strike'] = groupby_batter['プレイ前ストライク数'].diff().fillna(0) + all_pitch['pre_ball_foul']
    all_pitch['pre_foul_sum'] = all_pitch['打席内投球数'] - all_pitch['ball_count_sum']
    # 最後から何球目か
    groupby_bat_ball = all_pitch.groupby(['試合ID', 'イニング', 'イニング内打席数'], as_index=False)
    bat_ball_max = groupby_bat_ball['打席内投球数'].max()
    bat_ball_max.rename(columns={'打席内投球数': 'bat_ball_max'}, inplace=True)
    all_pitch = pd.merge(all_pitch, bat_ball_max, on=['試合ID', 'イニング', 'イニング内打席数'], how='left')
    all_pitch['last_ball'] = (all_pitch['bat_ball_max'] - all_pitch['打席内投球数'])/all_pitch['bat_ball_max']
    all_pitch.drop(columns=['bat_ball_max'], inplace=True)
    # ダミー変数
    # all_pitch = pd.get_dummies(all_pitch, columns=['ball_cnt'])
    # 不要な列を削除
    all_pitch.drop(columns=[
            '日付', '時刻', 
            '球場ID', '球場名', 
            '試合種別詳細', '表裏', 
            '投手投球左右', '投手役割', 
            '打者打席左右', '打者守備位置',
            '一塁走者ID', '二塁走者ID', '三塁走者ID', 
            '一塁手ID', '二塁手ID', '三塁手ID', '遊撃手ID', '左翼手ID', '中堅手ID', '右翼手ID', 
            '成績対象投手ID', '成績対象打者ID', 'home', 'top_bot'
        ], inplace=True)
    # 出力
    all_pitch.to_feather(OUTPUT)
    print(OUTPUT, all_pitch.shape)
    
    del all_pitch
