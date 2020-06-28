import os
import pandas as pd
import common

def ensemble(model_No, model_1, model_2, isBall=True, isCourse=True):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    if isBall:
        ball_1_csv = common.SUBMIT_BALL_CSV.format(model_1, model_1)
        ball_2_csv = common.SUBMIT_BALL_CSV.format(model_2, model_2)
        submit_ball_csv = common.SUBMIT_BALL_ENSMBL_CSV.format(model_No, model_1, model_2)

        ball_kind = ['straight', 'curve', 'slider', 'shoot', 'fork', 'changeup', 'sinker', 'cutball']
        ball_header = ['index'] + ball_kind

        ball_1 = pd.read_csv(ball_1_csv, names=ball_header)
        ball_2 = pd.read_csv(ball_2_csv, names=ball_header)
        print(ball_1.shape)
        print(ball_2.shape)

        ball_ensemble = pd.DataFrame(ball_1['index'])
        # 単純平均
        for ball in ball_kind:
            ball_ensemble[ball] = (ball_1[ball] + ball_2[ball])/2

        ball_ensemble.to_csv(submit_ball_csv, header=False, index=False)
        print(submit_ball_csv, ball_ensemble.shape)
        signate_command = 'signate submit --competition-id=275 ./{} --note ensemble_{}_{}'.format(submit_ball_csv, model_1, model_2)
        common.write_log(model_No, signate_command)

    if isCourse:
        course_1_csv = common.SUBMIT_COURSE_CSV.format(model_1, model_1)
        course_2_csv = common.SUBMIT_COURSE_CSV.format(model_2, model_2)
        submit_course_csv = common.SUBMIT_COURSE_ENSMBL_CSV.format(model_No, model_1, model_2)

        course_kind = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        course_header = ['index'] + course_kind

        course_1 = pd.read_csv(course_1_csv, names=course_header)
        course_2 = pd.read_csv(course_2_csv, names=course_header)
        print(course_1.shape)
        print(course_2.shape)

        course_ensemble = pd.DataFrame(course_1['index'])
        # 単純平均
        for course in course_kind:
            course_ensemble[course] = (course_1[course] + course_2[course])/2

        course_ensemble.to_csv(submit_course_csv, header=False, index=False)
        print(submit_course_csv, course_ensemble.shape)
        signate_command = 'signate submit --competition-id=276 ./{} --note ensemble_{}_{}'.format(submit_course_csv, model_1, model_2)
        common.write_log(model_No, signate_command)
    