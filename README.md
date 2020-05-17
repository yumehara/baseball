# SIGNATE ひろしまQuest2020

## 前処理

### 選手ごと

#### 1. Preprocess_pitch_2017_LR
 - 2017年の実績を集計 (1/4ずつ)
 - [投手成績1-4] `train_pitch` -> `intermediate/player/pit_2017_{}_{}.f`
 - [打者成績1-4] `train_pitch` -> `intermediate/player/bat_2017_{}_{}.f`

#### 2. Preprocess_player
 - 2017年の成績を選手情報にマージ (1/4ずつ)
 - [投手成績1-4] `intermediate/player/pit_2017_{}_{}.f` -> `intermediate/player/all_pitcher_{}_{}.f`
 - [打者成績1-4] `intermediate/player/bat_2017_{}_{}.f` -> `intermediate/player/all_player_{}_{}.f`

### 投球ごと

#### 3. Preprocess_pitch_agg
 - 2017年のボールカウントごとの投球情報を集計
 - [投球集計] `train_pitch` -> `intermediate/pitch/pitch_2017_{}.f`

#### 4. Preprocess_pitch
 - 投球情報に2017年の集計結果をマージ
 - [投球情報] `intermediate/pitch/pitch_2017_{}.f` -> `intermediate/pitch/all_pitch_{}.f`

### 全情報

#### 5. Preprocess_All
 - 選手情報と投球情報をマージ
##### サブモデルなしのとき
 - [投球情報] `intermediate/pitch/all_pitch_{}.f`
 - [投手情報1-4] `intermediate/player/all_pitcher_{}_{}.f`
 - [打者情報1-4] `intermediate/player/all_player_{}_{}.f`
 - [出力1-4] `intermediate/{}/all_merge_{}_{}.f`

##### サブモデルありのとき
 - [球種サブモデル1-4] `intermediate/{}/ball_predict_{}_{}.f`
 - [コースサブモデル1-4] `intermediate/{}/course_predict_{}_{}.f`
 - [出力1-4] `intermediate/{}/all_merge_{}_{}_sub.f`

## 学習

### 球種予測
#### BallPredict
##### サブモデルなしのとき
- [入力1-4] `intermediate/{}/all_merge_{}_{}.f`
- [中間出力1-4] `submit/{}/ball_{}_{}.f`
- [最終出力] `submit/{}/ball_{}.csv`
- [サブモデル1-4] `intermediate/{}/ball_predict_{}_{}.f`

##### サブモデルありのとき
- [入力1-4] `intermediate/{}/all_merge_{}_{}_sub.f`
- [中間出力1-4] `submit/{}/ball_{}_{}.f`
- [最終出力] `submit/{}/ball_{}_sub.csv`

### コース予測
#### CoursePredict
##### サブモデルなしのとき
- [入力1-4] `intermediate/{}/all_merge_{}_{}.f`
- [中間出力1-4] `submit/{}/course_{}_{}.f`
- [最終出力] `submit/{}/course_{}.csv`
- [サブモデル1-4] `intermediate/{}/course_predict_{}_{}.f`

##### サブモデルありのとき
- [入力1-4] `intermediate/{}/all_merge_{}_{}_sub.f`
- [中間出力1-4] `submit/{}/course_{}_{}.f`
- [最終出力] `submit/{}/course_{}_sub.csv`
