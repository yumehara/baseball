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


submit_No = '50'
metric = common.M_ERROR
boosting = common.DART


# # playerごと
# play2017.preprocess()
# player.preprocess(True)      # 穴埋めあり

# # 投球ごと
# ball.preprocess()
# pitch.preprocess()

# # 前処理
# use_sub_model = False
# merge.preprocess(submit_No, use_sub_model)
# print('--- preprocess ---')

# # コース予測サブモデル(LRHL)
# pred_course.train_predict2(submit_No, False, common.GBDT, metric, 'LR')
# pred_course.train_predict2(submit_No, False, common.GBDT, metric, 'HL')
pred_course.ensemble_RLHL(submit_No)
# print('--- predict course sub ---')

# コース予測
use_sub_model = True
pred_course.train_predict(submit_No, use_sub_model, boosting, metric)
print('--- predict course ---')

# # コース予測サブモデル(LRHL)
# pred_course.train_predict2(submit_No, False, boosting, metric, 'LR')
# pred_course.train_predict2(submit_No, False, boosting, metric, 'HL')
# pred_course.ensemble_RLHL(submit_No)
# print('--- predict course sub ---')

# # 前処理
# use_sub_model = True
# merge.preprocess(submit_No, use_sub_model)
# print('--- preprocess ---')

# 球種予測
use_sub_model = True
pred_ball.train_predict(submit_No, use_sub_model, boosting, metric)
print('--- predict ball ---')




# # アンサンブル(gbdt + dart)
# use_sub_model = True
# ensmbl.ensemble(submit_No, 47, 48, use_sub_model, True, False)

# Tuning
# python main.py 2>> tuning_0705.log
# use_sub_model = True
# pred_ball.tuning(submit_No, use_sub_model, boosting, metric)
# pred_course.tuning(submit_No, use_sub_model, boosting, metric)