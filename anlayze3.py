import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

feature_list = ['isrd_age_dcd', 'nur_hosp_yn', 'mdct_inu_rclss_dcd', 'blrs_cd',
                'dsas_acd_rst_dcd', 'kcd_gcd', 'urlb_fc_yn', 'fds_cust_yn',
                'smrtg_5y_passed_yn', 'mtad_cntr_yn', 'heltp_pf_ntyn', ]

binary_feature_list = ['nur_hosp_yn', 'fds_cust_yn', 'heltp_pf_ntyn',
                       'smrtg_5y_passed_yn', 'mtad_cntr_yn']

def my_model1(data):

    prob_list = [0, 0, 0]

# isrd_age_dcd: 고객나이구분코드            -> 코드 7일 때 target=2일 확률 79퍼센트  코드 9일 때 target=2일 확률 1퍼센트
    if data['isrd_age_dcd'] == 7:
        prob_list[2] += 0.79
    elif data['isrd_age_dcd'] == 9:
        prob_list[2] += -1

# fds_cust_yn: 보험사기이력고객여부          -> 코드 1일 때 target=0일 확률 4퍼센트
    if data['fds_cust_yn'] == 1:
        prob_list[1] += (1-0.04)*0.59
        prob_list[2] += (1-0.04)*0.16

# mtad_cntr_yn: 중도부가계약여부            -> 코드 1일 때 target=2일 확률 0.7퍼센트
    if data['mtad_cntr_yn'] == 1:
        prob_list[0] += (1-0.007)*0.25
        prob_list[1] += (1-0.007)*0.59

# heltp_pf_ntyn: 건강인우대계약여부          -> 코드 1일 때 target=2일 확률 0.9퍼센트
    if data['heltp_pf_ntyn'] == 1:
        prob_list[0] += (1-0.009)*0.25
        prob_list[1] += (1-0.009)*0.59

# prm_nvcd: 보험료구간코드                  -> 코드 99일 때 target=2일 확률 65퍼센트
#     if data['prm_nvcd'] == 99:
#         prob_list[2] += 0.65

# dsas_ltwt_gcd: 질병경중등급코드           -> 중증(코드1)이면 target=0일 확률 0퍼센트
    if data['dsas_ltwt_gcd'] == 1:
        prob_list[0] = -1

# kcd_gcd: KCD등급코드                     -> 신생물(암)(코드3), 내분비질환(코드5)이면 99퍼센트 target=0
# show_gragh('kcd_gcd')                   # -> 정신/신경질환(코드6,7)이면 target=0일 확률 3퍼센트로 매우 낮음
    if data['kcd_gcd'] == 3 or data['kcd_gcd'] == 5:
        prob_list[0] += -0.99
    elif data['kcd_gcd'] == 6 or data['kcd_gcd'] == 7:
        prob_list[0] += -0.97

# dsas_acd_rst_dcd: 질병구분코드            -> 당뇨병(코드16)은 target=0일 확률 0퍼센트대, 폐렴(코드11)의 경우 target=0일 확률 반정도 존재
# show_gragh('dsas_acd_rst_dcd')            # 그 외의 조건은 질병경중등급코드로 판별해도 무방하다고 봄

    if data['dsas_acd_rst_dcd'] == 16:
        prob_list[0] = -1

    # blrs_cd: 치료행위코드                     -> 진단만받은경우(코드0,3,4,5,6,7,11,12,13,14,15) -> target=0일 확률 0퍼센트에 가까움
    # show_gragh('blrs_cd')

    if data['blrs_cd'] in [0,3,4,5,6,7,11,12,13,14,15]:
        prob_list[0] = -1

    # bilg_isamt_s: 청구보험금                  -> index로 나와있어서 해석하기 힘듬. 단 index>2.5면 target=0일 확룰 0에 가까움
    # show_gragh('bilg_isamt_s')
    if data['bilg_isamt_s'] > 2.5:
        prob_list[0] = -1

    elif data['bilg_isamt_s'] in [86.0215, 64.5161, 53.7634]:
        prob_list[1] += 1


    # surop_blcnt_s: 수술청구건수               -> index > 12 이면 target=2일 확률 100에 가까움 index > 5 이면 target=0일 확률 0에 가까움
    # if data['surop_blcnt_s'] > 5:
    #     prob_list[0] += -1
    #
    #     if data['surop_blcnt_s'] > 12:
    #         prob_list[2] += 1

    # hspz_blcnt_s: 입원청구건수                  -> index > 6.7 이면 targe=0일 확률 0에 가까움
    if data['hspz_blcnt_s'] > 6.7:
        prob_list[0] += -1

    # 예외처리
    if prob_list == [0, 0, 0]:
        return None


    return prob_list.index(max(prob_list))

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



# feature들의 확률을 담고 있는 dict 생성
feature_prob_list = dict()
for feature in feature_list:
    feature_prob_list[feature] = get_portion(feature)

def multiply(list1, list2):
    tmp_list = []
    tmp_list.append(max(1, 100 * (list2[0]-0.25)))
    tmp_list.append(max(1, 100 * (list2[1] - 0.59)))
    tmp_list.append(max(1, 100 * (list2[2] - 0.12)))
    result_list = []
    result_list.append(list1[0] * tmp_list[0])
    result_list.append(list1[1] * tmp_list[1])
    result_list.append(list1[2] * tmp_list[2])
    return result_list

def make_target(data):
    prob_list = np.array([1, 1, 1])
    # feature들의 확률 곱해줌
    for feature in feature_list:
        if feature in binary_feature_list:
            prob_list = multiply(prob_list, feature_prob_list[feature][int(data[feature])])

        elif feature is 'kcd_gcd':
            prob_list = multiply(prob_list, feature_prob_list[feature][int(data[feature]-2)])

        elif feature is 'mdct_inu_rclss_dcd':
            if data[feature] == 9:
                prob_list=np.multiply(prob_list, feature_prob_list[feature][3])
            else:
                prob_list = np.multiply(prob_list, feature_prob_list[feature][int(data[feature]-1)])
        else:
            prob_list = np.multiply(prob_list, feature_prob_list[feature][int(data[feature]-1)])


    return np.argmax(prob_list)

result = pd.read_csv('test.csv')

# 1차분석 target list
target_list = []
for i in range(len(result)):
    an = my_model1(result.loc[i])
    target_list.append(an)
    print(str(i) + '번째 데이터 완료', 'target:' + str(an))

print('1차분석끝\n', target_list)
print(len(target_list))

# 2차분석 target list
target_list2 = []
for i in range(len(result)):
    an = make_target(result.loc[i])
    target_list2.append(an)
    print(str(i) + '번째 데이터 완료', 'target:' + str(an))

print('2차분석끝\n', target_list2)
print(len(target_list2))

# 1차, 2차 target list 합치기
for i in range(len(target_list)):
    if target_list[i] is None:
        target_list[i] = target_list2[i]

# 정확도 확인
# result['target2'] = target_list
# result_answer = result[result['target'] == result['target2']]
# print(len(target_list))
# print('정확도:', len(result_answer) / len(result))

# csv 파일 추출
result['target'] = target_list
result[['ID','target']].to_csv('result11.csv', index=False)


#printaw
print()