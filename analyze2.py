import pandas as pd
# 1이 가장 흔하기 때문에, 0과 2를 가장 잘 구분하는 피처들을 찾아보자.
# 정해지지 않는 애들은 비용이나 기간 등 정량적인 부분들이 가장 가까운 애한테로 가게한다.

# 전체 평균 target 0일때 25프로, 1일때 59프로 , 2일때 16프로
# 고객, 상품, 판매자 정보
# isrd_age_dcd: 고객나이구분코드
# fds_cust_yn: 보험사기이력고객여부
# smrtg_5y_passed_yn: 부담보5년경과여부
# mtad_cntr_yn: 중도부가계약여부
# heltp_pf_ntyn: 건강인우대계약여부
# prm_nvcd: 보험료구간코드
# inamt_nvcd: 가입금액구간코드
# ac_ctr_diff: 청구일계약일간기간구분코드
# ac_rst_diff: 청구일부활일간기간구분코드
# urlb_fc_yn: 부실판매자계약여부

# 질병 정보
# dsas_ltwt_gcd: 질병경중등급코드
# kcd_gcd: KCD등급코드
# dsas_acd_rst_dcd: 질병구분코드
# ar_rclss_cd: 발생지역구분코드
# blrs_cd: 치료행위코드
# mdct_inu_rclss_dcd: 의료기관구분코드
# nur_hosp_yn: 요양병원여부
# optt_nbtm_s: 통원횟수
# bilg_isamt_s: 청구보험금
# hspz_dys_s: 입원일수
# optt_blcnt_s: 통원청구건수
# surop_blcnt_s: 수술청구건수
# hspz_blcnt_s: 입원청구건수
# 금액관련

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


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#
# my_model을 이용한 분류율
df = pd.read_csv('test.csv')
list = [my_model1(df.loc[i]) for i in range(len(df))]
df['target'] = list
df_null = df[df.isnull().any(1)]
print('분류율:',1 - (len(df_null)/len(df)))

# train_set으로 정확도 확인
df = pd.read_csv('train.csv').head(10000)
list = [my_model1(df.loc[i]) for i in range(len(df))]
df['target2'] = list
df = df.dropna()
result = df[df['target'] == df['target2']]
print('정확도:', len(result)/len(df))

# 결측값 다 1로 바꾸기
# df = pd.read_csv('test.csv')
# list = [my_model(df.loc[i]) for i in range(len(df))]
# df['target'] = list
# df_full = df.fillna('1')
# df_full[['ID','target']].to_csv('result4.csv', index=False)

# # 결과 제출용 test_set
# test_set = pd.read_csv('test.csv')
# list = []
# for i in range(len(test_set)):
#     list.append(my_model(test_set.loc[i]))
# test_set['target'] = list
# test_set[['ID','target']].to_csv('result3.csv', index=False)
#
# #7035, 1310 / 22071


# # 결측값 analyze1로 수정하기
# raw_data = pd.read_csv('train.csv')
# count = raw_data.count()
#
# # 질병검증코드 1~5의 단계에서(4,5단계는 tran.csv에 없음) 1 -> 3 으로 갈수록 target이 0일 확률이 높음
# target_mean_for_dsas_ltwt_gcd = raw_data['target'].groupby(raw_data['dsas_ltwt_gcd']).mean()
#
# # KCD 질병 분류표 1번 누락/ 2 4 8 10 11 13 14 번이 수치 낮게 나옴
# target_mean_for_kcd_gcd = raw_data['target'].groupby(raw_data['kcd_gcd']).mean()
#
# # 질병구분코드
# target_mean_for_dsas_acd_rst_dcd = raw_data['target'].groupby(raw_data['dsas_acd_rst_dcd']).mean()
#
# # 발생지역구분코드
# target_mean_for_ar_rclss_cd = raw_data['target'].groupby(raw_data['ar_rclss_cd']).mean()
#
# # 치료행위코드
# target_mean_for_blrs_cd = raw_data['target'].groupby(raw_data['blrs_cd']).mean()
#
#
# def my_model2(data):
#     weight_of_dsas_ltwt_gcd = target_mean_for_dsas_ltwt_gcd[data['dsas_ltwt_gcd']]  # 1~5 등급 중 4,5누락이고 1, 2, 3 중 선택
#
#     weight_of_kcd_gcd = target_mean_for_kcd_gcd[data['kcd_gcd']]                    # 1 ~18 등급 중 1, 15, 16, 17, 18 누락
#
#     weight_of_dsas_acd_rst_dcd = target_mean_for_dsas_acd_rst_dcd[data['dsas_acd_rst_dcd']]
#
#     weight_of_ar_rclss_cd = target_mean_for_ar_rclss_cd[data['ar_rclss_cd']]
#
#     return weight_of_ar_rclss_cd + weight_of_dsas_acd_rst_dcd + weight_of_dsas_ltwt_gcd + weight_of_kcd_gcd
#
# def cal_target(data):
#     if data >= 4.619529370414207:
#         return 2
#     elif data >= 2.722750487764433 and data <4.619529370414207:
#         return 1
#     else:
#         return 0
#
# df = pd.read_csv('test.csv')
# scores = [my_model2(df.loc[i]) for i in range(len(df))]
# targets = [cal_target(i) for i in scores]
#
# list = [my_model(df.loc[i]) for i in range(len(df))]
#
# for i in range(len(list)):
#     if list[i] is None:
#         list[i] = targets[i]
#
# df['target'] = list
# df[['ID','target']].to_csv('result5.csv', index=False)