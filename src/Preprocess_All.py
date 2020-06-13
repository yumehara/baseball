# coding:utf-8
import os
import gc
import pandas as pd
import feather
import common

def preprocess(model_No, use_sub_model, is_Ball):

    # 出力先のフォルダ作成
    os.makedirs(common.OUTPUT_PATH.format(model_No), exist_ok=True)

    for sample_No in range(1, common.DIVIDE_NUM+1):
        ALL_PITCH = common.ALL_PITCH
        ALL_PITCHER = common.ALLPITCHER.format(sample_No)
        ALL_CATCHER = common.ALLCATCHER.format(sample_No)
        ALL_PLAYER = common.ALLPLAYER.format(sample_No)

        SUB_BALL = common.PREDICT_BALL.format(model_No, model_No, sample_No)
        SUB_COURSE = common.PREDICT_COURSE.format(model_No, model_No, sample_No)

        OUTPUT = common.ALL_MERGE.format(model_No, model_No, sample_No)
        OUTPUT_SUB = common.ALL_MERGE_SUB.format(model_No, model_No, sample_No)

        # 投球情報
        all_pitch = pd.read_feather(ALL_PITCH)
        print(all_pitch.shape)

        # 投手情報
        all_pitcher = pd.read_feather(ALL_PITCHER)
        print(all_pitcher.shape)

        # 捕手情報
        all_catcher = pd.read_feather(ALL_CATCHER)
        print(all_catcher.shape)

        # 打者情報
        all_player = pd.read_feather(ALL_PLAYER)
        print(all_player.shape)

        # Join
        merge_all = pd.merge(all_pitch, all_pitcher, left_on=['投手ID', '年度', 'pit_bat'], right_on=['選手ID', '年度', 'pit_bat'], how='left')
        merge_all = pd.merge(merge_all, all_player, left_on=['打者ID', '年度'], right_on=['選手ID', '年度'], how='left', suffixes=['_pit', '_bat'])
        merge_all = pd.merge(merge_all, all_catcher, left_on=['捕手ID', '年度', 'pit_bat'], right_on=['選手ID', '年度', 'pit_bat'], how='left', suffixes=['', '_cat'])

        del all_pitch, all_pitcher, all_player

        # player同士の組み合わせ
        merge_all['salary_dif_p-b'] = merge_all['salary_pit'] - merge_all['salary_bat']
        merge_all['play_year_dif_p-b'] = merge_all['play_year_pit'] - merge_all['play_year_bat']
        merge_all['age_dif_p-b'] = merge_all['age_pit'] - merge_all['age_bat']
        merge_all['salary_year_dif_p-b'] = merge_all['salary_year_pit'] - merge_all['salary_year_bat']
        merge_all['salary_x_year_dif_p-b'] = merge_all['salary_x_year_pit'] - merge_all['salary_x_year_bat']
        merge_all['rank_year_dif_p-b'] = merge_all['rank_year_pit'] - merge_all['rank_year_bat']
        merge_all['rank_x_year_dif_p-b'] = merge_all['rank_x_year_pit'] - merge_all['rank_x_year_bat']
        merge_all['bmi_dif_p-b'] = merge_all['bmi_pit'] - merge_all['bmi_bat']

        merge_all['salary_dif_p-c'] = merge_all['salary_pit'] - merge_all['salary']
        merge_all['play_year_dif_p-c'] = merge_all['play_year_pit'] - merge_all['play_year']
        merge_all['age_dif_p-c'] = merge_all['age_pit'] - merge_all['age']
        merge_all['salary_year_dif_p-c'] = merge_all['salary_year_pit'] - merge_all['salary_year']
        merge_all['salary_x_year_dif_p-c'] = merge_all['salary_x_year_pit'] - merge_all['salary_x_year']
        merge_all['rank_year_dif_p-c'] = merge_all['rank_year_pit'] - merge_all['rank_year']
        merge_all['rank_x_year_dif_p-c'] = merge_all['rank_x_year_pit'] - merge_all['rank_x_year']
        merge_all['bmi_dif_p-c'] = merge_all['bmi_pit'] - merge_all['bmi']

        merge_all['salary_dif_b-c'] = merge_all['salary_bat'] - merge_all['salary']
        merge_all['play_year_dif_b-c'] = merge_all['play_year_bat'] - merge_all['play_year']
        merge_all['age_dif_b-c'] = merge_all['age_bat'] - merge_all['age']
        merge_all['salary_year_dif_b-c'] = merge_all['salary_year_bat'] - merge_all['salary_year']
        merge_all['salary_x_year_dif_b-c'] = merge_all['salary_x_year_bat'] - merge_all['salary_x_year']
        merge_all['rank_year_dif_b-c'] = merge_all['rank_year_bat'] - merge_all['rank_year']
        merge_all['rank_x_year_dif_b-c'] = merge_all['rank_x_year_bat'] - merge_all['rank_x_year']
        merge_all['bmi_dif_b-c'] = merge_all['bmi_bat'] - merge_all['bmi']

        # 球種の組合せ
        ball_kind = ['straight', 'curve', 'slider', 'shoot', 'fork', 'changeup', 'sinker', 'cutball']
        
        for ball in ball_kind:
            target = 'sub_' + ball
            bc_src = 'bc_' + ball
            merge_all[target] = merge_all[bc_src] - merge_all[ball]

        for ball in ball_kind:
            target = 'div_' + ball
            bc_src = 'bc_' + ball
            merge_all[target] = merge_all[bc_src] / merge_all[ball]

        for ball in ball_kind:
            target = 'mul_' + ball
            bc_src = 'bc_' + ball
            merge_all[target] = merge_all[bc_src] * merge_all[ball]

        for ball in ball_kind:
            target = 'ave_' + ball
            bc_src = 'bc_' + ball
            merge_all[target] = (merge_all[bc_src] + merge_all[ball])/2

        for ball in ball_kind:
            target = 'rate_' + ball
            ave_src = 'ave_' + ball
            merge_all[target] = merge_all[ave_src] /(1-merge_all['ave_straight'])
    
        ball_kind_bc = list(map(lambda x: 'bc_' + x, ball_kind))
        merge_all.drop(columns=ball_kind, inplace=True)
        merge_all.drop(columns=ball_kind_bc, inplace=True)

        # コースの組合せ
        course_kind = ['course_0', 'course_1', 'course_2', 'course_3', 'course_4', 'course_5', 'course_6', 
                'course_7', 'course_8', 'course_9', 'course_10', 'course_11', 'course_12',
               'high_str', 'high_ball', 'mid_str', 'low_str', 'low_ball', 
               'left_str', 'left_ball', 'center_str', 'right_str', 'right_ball']

        for course in course_kind:
            target = 'sub_' + course
            bc_src = 'bc_' + course
            merge_all[target] = merge_all[bc_src] - merge_all[course]

        for course in course_kind:
            target = 'div_' + course
            bc_src = 'bc_' + course
            merge_all[target] = merge_all[bc_src] / merge_all[course]

        for course in course_kind:
            target = 'mul_' + course
            bc_src = 'bc_' + course
            merge_all[target] = merge_all[bc_src] * merge_all[course]

        for course in course_kind:
            target = 'ave_' + course
            bc_src = 'bc_' + course
            merge_all[target] = (merge_all[bc_src] + merge_all[course])/2

        course_kind_bc = list(map(lambda x: 'bc_' + x, course_kind))
        merge_all.drop(columns=course_kind, inplace=True)
        merge_all.drop(columns=course_kind_bc, inplace=True)

        # ダミー変数
        merge_all = pd.get_dummies(merge_all, columns=['pit_bat'])

        # 不要な列を削除
        merge_all.drop(columns=[
            '選手ID_pit', '選手ID_bat', '選手ID',
            '試合内連番',
            '年度', 
            '試合ID', 
            'ホームチームID', 'アウェイチームID', 
            '投手ID', '投手チームID', 
            '打者ID', '打者チームID', 
            'プレイ前走者状況', 
            '捕手ID', 
            'opening_date', 'game_date',
            'start_time', 'game_time', 'elapsed_time'
        ], inplace=True)

        # Rename
        merge_all.rename(columns={
            'データ内連番': 'No',
            '試合内投球数': 'pitch_cnt_in_game',
            'イニング': 'inning',
            'イニング内打席数': 'bat_cnt_in_inning',
            '打席内投球数': 'pitch_cnt_in_bat',
            '投手登板順': 'pitch_order',
            '投手試合内対戦打者数': 'player_cnt_in_game',
            '投手試合内投球数': 'pitcher_cnt_in_game',
            '投手イニング内投球数': 'pitcher_cnt_in_inning',
            '打者打順': 'bat_order',
            '打者試合内打席数': 'bat_cnt_in_game',
            'プレイ前ホームチーム得点数': 'home_point',
            'プレイ前アウェイチーム得点数': 'away_point',
            'プレイ前アウト数': 'out_cnt',
            'プレイ前ボール数': 'ball_cnt',
            'プレイ前ストライク数': 'strike_cnt',
        }, inplace=True)

        print(merge_all.shape)

        # 出力
        if not use_sub_model:
            # 出力(sub-modelなし)
            merge_all.to_feather(OUTPUT)
            print(OUTPUT, merge_all.shape)
        else:
            if is_Ball:
                merge_sub = pd.read_feather(SUB_COURSE)
            else:
                merge_sub = pd.read_feather(SUB_BALL)
            
            # 予測結果を特徴量に加える
            merge_sub = pd.concat([merge_all, merge_sub], axis=1)
            print(merge_sub.shape)
            # 出力(sub-modelあり)
            merge_sub.to_feather(OUTPUT_SUB)
            print(OUTPUT_SUB, merge_sub.shape)
            del merge_sub

        del merge_all
    
    gc.collect()
