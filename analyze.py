import pandas as pd

raw_data = pd.read_csv('train.csv')
count = raw_data.count()
print(count)
# 0 -> 93793  25 퍼센트
# 1 -> 226036 59 퍼센트
# 전체 -> 377928

# print(raw_data.head())

# 질병검증코드 1~5의 단계에서(4,5단계는 tran.csv에 없음) 1 -> 3 으로 갈수록 target이 0일 확률이 높음
target_mean_for_dsas_ltwt_gcd = raw_data['target'].groupby(raw_data['dsas_ltwt_gcd']).mean()

# KCD 질병 분류표 1번 누락/ 2 4 8 10 11 13 14 번이 수치 낮게 나옴
target_mean_for_kcd_gcd = raw_data['target'].groupby(raw_data['kcd_gcd']).mean()

# 질병구분코드
target_mean_for_dsas_acd_rst_dcd = raw_data['target'].groupby(raw_data['dsas_acd_rst_dcd']).mean()

# 발생지역구분코드
target_mean_for_ar_rclss_cd = raw_data['target'].groupby(raw_data['ar_rclss_cd']).mean()

# 치료행위코드
target_mean_for_blrs_cd = raw_data['target'].groupby(raw_data['blrs_cd']).mean()


def my_model(data):
    weight_of_dsas_ltwt_gcd = target_mean_for_dsas_ltwt_gcd[data['dsas_ltwt_gcd']]  # 1~5 등급 중 4,5누락이고 1, 2, 3 중 선택

    weight_of_kcd_gcd = target_mean_for_kcd_gcd[data['kcd_gcd']]                    # 1 ~18 등급 중 1, 15, 16, 17, 18 누락

    weight_of_dsas_acd_rst_dcd = target_mean_for_dsas_acd_rst_dcd[data['dsas_acd_rst_dcd']]

    weight_of_ar_rclss_cd = target_mean_for_ar_rclss_cd[data['ar_rclss_cd']]

    return weight_of_ar_rclss_cd + weight_of_dsas_acd_rst_dcd + weight_of_dsas_ltwt_gcd + weight_of_kcd_gcd

def cal_target(data):
    if data >= 4.619529370414207:
        return 2
    elif data >= 2.722750487764433 and data <4.619529370414207:
        return 1
    else:
        return 0

test_set = pd.read_csv('test.csv')
print(test_set)

list = []
for i in range(len(test_set)):
    list.append(my_model(test_set.loc[i]))

test_set['score'] = list

list = []
for i in range(len(test_set)):
    list.append(cal_target(test_set['score'].loc[i]))

test_set['target'] = list

print(test_set['target'])

test_set[['ID','target']].to_csv('result2.csv', index=False)
