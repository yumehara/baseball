# coding:utf-8
import os
import Preprocess_All as merge
import Preprocess_ball as ball
import Preprocess_pitch as pitch
import Preprocess_player_2017 as play2017
import Preprocess_player as player
import Predict_Ball as pred_ball
import Predict_Course as pred_course
import ensemble as ensmbl
import common


submit_No = '44'
use_sub_model = False
metric = common.M_ERROR
boosting = common.DART

# # playerごと
# play2017.preprocess()
# player.preprocess(True)      # 穴埋めあり

# # 投球ごと
# ball.preprocess()
# pitch.preprocess()

# # 前処理
# merge.preprocess(submit_No, use_sub_model, True)
# print('--- preprocess ---')

# # 球種予測
# pred_ball.train_predict(submit_No, use_sub_model, boosting, metric)
# print('--- predict ball w/o sub ---')

# # コース予測
# pred_course.train_predict(submit_No, use_sub_model, boosting, metric)
# print('--- predict course w/o sub ---')

# アンサンブル(gbdt + dart)
ensmbl.ensemble(submit_No, 42, 43)