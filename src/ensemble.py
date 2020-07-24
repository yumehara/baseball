import os
import pandas as pd
import common
import numpy as np

# 幾何平均
geometric_ave = True

def ensemble(model_No, sub_str_1, sub_str_2, isBall, cv):

    # 出力先のフォルダ作成
    os.makedirs(common.SUBMIT_PATH.format(model_No), exist_ok=True)

    if isBall:
        ball_1_csv = common.SUBMIT_BALL_CSV.format(model_No, model_No, sub_str_1)
        ball_2_csv = common.SUBMIT_BALL_CSV.format(model_No, model_No, sub_str_2)
        submit_ball_csv = common.SUBMIT_BALL_ENSMBL_CSV.format(model_No, model_No, sub_str_1, sub_str_2)

        ball_kind = ['straight', 'curve', 'slider', 'shoot', 'fork', 'changeup', 'sinker', 'cutball']
        ball_header = ['index'] + ball_kind

        ball_1 = pd.read_csv(ball_1_csv, names=ball_header)
        ball_2 = pd.read_csv(ball_2_csv, names=ball_header)
        print(ball_1.shape)
        print(ball_2.shape)

        ball_ensemble = pd.DataFrame(ball_1['index'])
        
        if geometric_ave:   # 幾何平均
            for ball in ball_kind:
                ball_ensemble[ball] = np.sqrt(ball_1[ball] * ball_2[ball])
            ball_ensemble['sum'] = ball_ensemble[ball_kind].sum(axis=1)
            for ball in ball_kind:
                ball_ensemble[ball] = ball_ensemble[ball]/ball_ensemble['sum']
            ball_ensemble.drop(columns=['sum'], inplace=True)
            print('geometric_ave')
        else:               # 単純平均
            for ball in ball_kind:
                ball_ensemble[ball] = (ball_1[ball] + ball_2[ball])/2
            print('simple_ave')

        ball_ensemble.to_csv(submit_ball_csv, header=False, index=False)
        print(submit_ball_csv, ball_ensemble.shape)
        signate_command = 'signate submit --competition-id=275 ./{} --note ensemble_{}'.format(submit_ball_csv, cv)
        common.write_log(model_No, signate_command)

    else:
        course_1_csv = common.SUBMIT_COURSE_CSV.format(model_No, model_No, sub_str_1)
        course_2_csv = common.SUBMIT_COURSE_CSV.format(model_No, model_No, sub_str_2)
        submit_course_csv = common.SUBMIT_COURSE_ENSMBL_CSV.format(model_No, model_No, sub_str_1, sub_str_2)

        course_kind = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        course_header = ['index'] + course_kind

        course_1 = pd.read_csv(course_1_csv, names=course_header)
        course_2 = pd.read_csv(course_2_csv, names=course_header)
        print(course_1.shape)
        print(course_2.shape)

        course_ensemble = pd.DataFrame(course_1['index'])
        
        if geometric_ave:   # 幾何平均
            for course in course_kind:
                course_ensemble[course] = np.sqrt(course_1[course] * course_2[course])
            course_ensemble['sum'] = course_ensemble[course_kind].sum(axis=1)
            for course in course_kind:
                course_ensemble[course] = course_ensemble[course]/course_ensemble['sum']
            course_ensemble.drop(columns=['sum'], inplace=True)
            print('geometric_ave')
        else:               # 単純平均
            for course in course_kind:
                course_ensemble[course] = (course_1[course] + course_2[course])/2
            print('simple_ave')

        course_ensemble.to_csv(submit_course_csv, header=False, index=False)
        print(submit_course_csv, course_ensemble.shape)
        signate_command = 'signate submit --competition-id=276 ./{} --note ensemble_{}'.format(submit_course_csv, cv)
        common.write_log(model_No, signate_command)
    