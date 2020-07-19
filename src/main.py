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


submit_No = '53'
metric = common.M_ERROR

boosting0 = common.GBDT
boosting1 = common.DART
boosting2 = common.GBDT

# # playerごと
# play2017.preprocess()
# player.preprocess(True)      # 穴埋めあり

# # 投球ごと
# ball.preprocess()
# pitch.preprocess()

# # 前処理
# merge.preprocess(submit_No)
# print('--- preprocess ---')

# # コース予測サブモデル(LRHL)
# pred_course.train_predict2(submit_No, boosting0, metric, 'LR')
# pred_course.train_predict2(submit_No, boosting0, metric, 'HL')
# pred_course.ensemble_RLHL(submit_No)
# print('--- predict course sub ---')

# # コース予測(1:dart)
# use_sub_model = False
# sub_str1 = boosting1 + '_' + metric
# pred_course.train_predict(submit_No, use_sub_model, boosting1, metric, sub_str1)
# print('--- predict course {}---'.format(sub_str1))

# # コース予測(2:gbdt)
# sub_str2 = boosting2 + '_' + metric
# pred_course.train_predict(submit_No, use_sub_model, boosting2, metric, sub_str2)
# print('--- predict course {}---'.format(sub_str2))


# 球種予測(1:dart)
use_sub_model = True
sub_str1 = boosting1 + '_' + metric
pred_ball.train_predict(submit_No, use_sub_model, boosting1, metric, sub_str1)
print('--- predict ball {}---'.format(sub_str1))

# 球種予測(2:gbdt)
sub_str2 = boosting2 + '_' + metric
pred_ball.train_predict(submit_No, use_sub_model, boosting2, metric, sub_str2)
print('--- predict ball {}---'.format(sub_str2))


# アンサンブル(gbdt + dart)
ensmbl.ensemble(submit_No, sub_str1, sub_str2, True, False)


# Tuning
# python main.py 2>> tuning_0705.log
# use_sub_model = True
# pred_ball.tuning(submit_No, use_sub_model, boosting, metric)
# pred_course.tuning(submit_No, use_sub_model, boosting, metric)