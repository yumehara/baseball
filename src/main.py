# coding:utf-8
import os
import Preprocess_All as merge
import Preprocess_ball as ball
import Preprocess_pitch as pitch
import Preprocess_player_2017 as play2017
import Preprocess_player as player
import Predict_Ball as pred_ball
import Predict_Course as pred_course


submit_No = "32"

# playerごと
play2017.preprocess()
player.preprocess(True)      # 穴埋めあり

# 投球ごと
ball.preprocess()
pitch.preprocess()

# サブモデルなしコース予測用 前処理
use_sub_model = False
merge.preprocess(submit_No, use_sub_model, False)
print('--- preprocess for course ---')

# サブモデルなしコース予測
pred_course.train_predict(submit_No, use_sub_model)
print('--- predict course w/o sub ---')

# サブモデルあり球種予測用 前処理
use_sub_model = True
merge.preprocess(submit_No, use_sub_model, True)
print('--- preprocess for ball ---')

# サブモデルあり球種予測
pred_ball.train_predict(submit_No, use_sub_model)
print('--- predict ball w/ sub ---')
