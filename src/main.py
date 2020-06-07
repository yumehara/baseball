# coding:utf-8
import os
import Preprocess_All as merge
import Preprocess_ball as ball2017
import Preprocess_ball_2018 as ball2018
import Preprocess_pitch as pitch
import Preprocess_player_2017 as play2017
import Preprocess_player_2018 as play2018
import Preprocess_player as player
import Predict_Ball as pred_ball
import Predict_Ball_2018 as pred_ball2
import Predict_Course as pred_course


submit_No = "30"

# playerごと (2017年)
# play2017.preprocess()
# player.preprocess(True, True)      # 穴埋めあり

# # 投球ごと (2017年)
# ball2017.preprocess()
# pitch.preprocess()

# # サブモデルなしコース予測用 前処理
# use_sub_model = False
# merge.preprocess(submit_No, use_sub_model, False)
# print('--- preprocess for course ---')

# # サブモデルなしコース予測
# pred_course.train_predict(submit_No, use_sub_model)
# print('--- predict course w/o sub ---')

# # サブモデルあり球種予測用 前処理
# use_sub_model = True
# merge.preprocess(submit_No, use_sub_model, True)
# print('--- preprocess for ball ---')

# # サブモデルあり球種予測
# pred_ball.train_predict(submit_No, use_sub_model)
# print('--- predict ball w/ sub ---')



# 2018-2019の予測結果の集計
play2018.preprocess(submit_No)
# playerごと (2018-2019年)
player.preprocess(True, False)      # 穴埋めあり

# 投球ごと (2018-2019年)
ball2018.preprocess(submit_No)
pitch.preprocess(False)

# サブモデルなし球種予測用 前処理(2018-2019年)
use_sub_model = False
merge.preprocess(submit_No, use_sub_model, True, False)
print('--- preprocess for ball (2018)---')

# サブモデルなし球種予測(2018-2019年)
use_sub_model = False
pred_ball2.train_predict(submit_No, use_sub_model)
print('--- predict ball w/o sub (2018)---')