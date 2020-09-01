import pandas as pd
import numpy as np
from time import time
# fds_cust_yn: 보험사기이력고객여부          -> 코드 1일 때 target=0일 확률 4퍼센트
# show_gragh('fds_cust_yn')

# smrtg_5y_passed_yn: 부담보5년경과여부     -> 무의미
#show_gragh('smrtg_5y_passed_yn')

# mtad_cntr_yn: 중도부가계약여부            -> 코드 1일 때 target=2일 확률 0.7퍼센트
# show_gragh('mtad_cntr_yn')

# heltp_pf_ntyn: 건강인우대계약여부          -> 코드 1일 때 target=2일 확률 0.9퍼센트
# show_gragh('heltp_pf_ntyn')

# urlb_fc_yn: 부실판매자계약여부             -> 의외로 효과 없음
# show_gragh('urlb_fc_yn')

# kcd_gcd: KCD등급코드                     -> 신생물(암)(코드3), 내분비질환(코드5)이면 target=0일 확률 0
# show_gragh('kcd_gcd')                   # -> 정신/신경질환(코드6,7)이면 target=0일 확률 3퍼센트로 매우 낮음

# dsas_acd_rst_dcd: 질병구분코드            -> 당뇨병(코드16)은 target=0일 확률 0퍼센트대, 폐렴(코드11)의 경우 target=0일 확률 반정도 존재
# show_gragh('dsas_acd_rst_dcd')            # 그 외의 조건은 질병경중등급코드로 판별해도 무방하다고 봄

# blrs_cd: 치료행위코드                     -> 진단만받은경우(코드0,3,4,5,6,7,11,12,13,14,15) -> target=0일 확률 0퍼센트에 가까움
# show_gragh('blrs_cd')

# mdct_inu_rclss_dcd: 의료기관구분코드
# show_gragh('mdct_inu_rclss_dcd')

# nur_hosp_yn: 요양병원여부
# show_gragh('nur_hosp_yn')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('train.csv')


def get_portion(feature):
    pdf1 = pd.pivot_table(df[['ID', 'target', feature]],  # 피벗할 데이터프레임
                          index=feature,  # 행 위치에 들어갈 열
                          columns='target',  # 열 위치에 들어갈 열
                          values='ID',    # 데이터로 사용할 열
                          aggfunc=['count'])  # 데이터 집계함수
    pdf1 = pdf1.to_numpy(dtype=np.float32)
    np.nan_to_num(pdf1, copy=False)
    sum = np.zeros(len(pdf1))

    # 각 행의 합 구하기
    for i in range(len(pdf1)):
        for j in range(len(pdf1[0])):
            sum[i] += pdf1[i][j]

    # 각 성분을 비율로 나타내기
    for i in range(len(pdf1)):
        for j in range(len(pdf1[0])):
            pdf1[i][j] = round(pdf1[i][j]/(sum[i]),2)
    return pdf1

def list_multiply(list1, list2):
    for i in range(len(list1)):
        list1[i] = list1[i] * list2[i]
    return list1

feature_list = ['isrd_age_dcd', 'nur_hosp_yn', 'mdct_inu_rclss_dcd', 'blrs_cd',
                          'dsas_acd_rst_dcd', 'kcd_gcd', 'urlb_fc_yn', 'fds_cust_yn',
                          'smrtg_5y_passed_yn', 'mtad_cntr_yn', 'heltp_pf_ntyn', ]

binary_feature_list = ['nur_hosp_yn', 'fds_cust_yn', 'heltp_pf_ntyn',
                           'smrtg_5y_passed_yn', 'mtad_cntr_yn']

# feature들의 확률을 담고 있는 dict 생성
feature_prob_list = dict()
for feature in feature_list:
    feature_prob_list[feature] = get_portion(feature)

def make_target(data):
    b_time = time()
    prob_list = np.array([15000, 10, 20000000])
    print('feature 들의 확률을 담고 있는 dict 생성:', str(time()-b_time)+'초 소요')
# 연령: isrd_age_dcd , 요양병원부:nur_hosp_yn, 보험사기이력:fds_cust_yn , 건강인우대게약여부: heltp_pf_ntyn
# 부담보5년경과여부: smrtg_5y_passed_yn, 중도부가계약여부: mtad_cntr_yn
#
    b_time = time()
    for feature in feature_list:
        if feature in binary_feature_list:
            prob_list = np.multiply(prob_list, feature_prob_list[feature][data[feature]])
        elif feature is 'kcd_gcd':
            prob_list = np.multiply(prob_list, feature_prob_list[feature][data[feature]-2])
        else:
            prob_list = np.multiply(prob_list, feature_prob_list[feature][data[feature]-1])
    print('연산 수행:', str(time()-b_time) + '초 소요')

    print(prob_list.round(4))
    return np.argmax(prob_list)


result = pd.read_csv('train.csv')[['isrd_age_dcd', 'nur_hosp_yn', 'mdct_inu_rclss_dcd', 'blrs_cd',
                            'dsas_acd_rst_dcd', 'kcd_gcd', 'urlb_fc_yn', 'fds_cust_yn',
                            'smrtg_5y_passed_yn', 'mtad_cntr_yn', 'heltp_pf_ntyn', 'target' ]].head(100)

target_list = []

for i in range(len(result)):
    an = make_target(result.loc[i])
    target_list.append(an)
    print(str(i) + '번째 데이터 완료', 'target:' + str(an))

result['target2'] = target_list
result_answer = result[result['target'] == result['target2']]
print('정확도:', len(result_answer) / len(result))