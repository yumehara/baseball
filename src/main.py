# coding:utf-8
import os
import Preprocess_All as all
import Preprocess_ball as ball
import Preprocess_pitch as pitch
import Preprocess_player_2017 as play2017
import Preprocess_player as player
import Predict_Ball as pred_b


# play2017.preprocess_player()
# player.preprocess_player()

# ball.preprocess_ball()
# pitch.preprocess_pitch()

#all.preprocess_all(22, False)

print('preprocess_All complete')

pred_b.Train_Ball(22, False)

print('predict complete')