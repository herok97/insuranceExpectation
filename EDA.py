import plotly.express as px
import tensorflow as tf
import matplotlib.pyplot as plt
# 1이 가장 흔하기 때문에, 0과 2를 가장 잘 구분하는 피처들을 찾아보자.
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

# hsp_avg_hspz_bilg_isamt_s: 병원별평균입원청구보험금
# hsp_avg_optt_bilg_isamt_s: 병원별평균통원청구보험금
# hsp_avg_surop_bilg_isamt_s: 병원별평균수술청구보험금
# hsp_avg_diag_bilg_isamt_s: 병원별평균진단청구보험금

# ID,,,hsp_avg_optt_bilg_isamt_s,hsp_avg_surop_bilg_isamt_s,,,,hsp_avg_diag_bilg_isamt_s,,,
# dsas_avg_diag_bilg_isamt_s,,base_ym,,hsp_avg_hspz_bilg_isamt_s,,mtad_cntr_yn,,,,,dsas_avg_optt_bilg_isamt_s,
# ,,dsas_avg_surop_bilg_isamt_s,,dsas_avg_hspz_bilg_isamt_s,,,,,target

import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')

def show_gragh(code):
    data[[code, 'target']].value_counts().to_csv(code + '.csv')
    df = pd.read_csv(code + '.csv')
    fig = px.scatter(df, x= code, y='target', color='target', size='0', size_max=50)
    fig.show()





# isrd_age_dcd: 고객나이구분코드            -> 코드 7일 때 target=2일 확률 79퍼센트   코드 9일 때 target=2일 확률 1퍼센트
show_gragh('isrd_age_dcd')

# fds_cust_yn: 보험사기이력고객여부          -> 코드 1일 때 target=0일 확률 4퍼센트
# show_gragh('fds_cust_yn')

# smrtg_5y_passed_yn: 부담보5년경과여부     -> 무의미
#show_gragh('smrtg_5y_passed_yn')

# mtad_cntr_yn: 중도부가계약여부            -> 코드 1일 때 target=2일 확률 0.7퍼센트
# show_gragh('mtad_cntr_yn')

# heltp_pf_ntyn: 건강인우대계약여부          -> 코드 1일 때 target=2일 확률 0.9퍼센트
# show_gragh('heltp_pf_ntyn')

# prm_nvcd: 보험료구간코드                  -> 코드 99일 때 target=2일 확률 65퍼센트
# show_gragh('prm_nvcd')

# inamt_nvcd: 가입금액구간코드               -> 위와 같음 근데 unknown임
# show_gragh('inamt_nvcd')

# ac_ctr_diff: 청구일계약일간기간구분코드      -> 위와 같음 근데 unknown임 // 위의 두개와 이 자료는 빼는게 나을듯
# show_gragh('ac_ctr_diff')

# ac_rst_diff: 청구일부활일간기간구분코드      -> 얘도 버려도 될듯
# show_gragh('ac_rst_diff')

# urlb_fc_yn: 부실판매자계약여부             -> 의외로 효과 없음
# show_gragh('urlb_fc_yn')


############################################################################
# 질병 정보
# dsas_ltwt_gcd: 질병경중등급코드           -> 중증(코드1)이면 target=0일 확률 0퍼센트
# show_gragh('dsas_ltwt_gcd')

# kcd_gcd: KCD등급코드                     -> 신생물(암)(코드3), 내분비질환(코드5)이면 target=0일 확률 0
# show_gragh('kcd_gcd')                   # -> 정신/신경질환(코드6,7)이면 target=0일 확률 3퍼센트로 매우 낮음

# dsas_acd_rst_dcd: 질병구분코드            -> 당뇨병(코드16)은 target=0일 확률 0퍼센트대, 폐렴(코드11)의 경우 target=0일 확률 반정도 존재
# show_gragh('dsas_acd_rst_dcd')            # 그 외의 조건은 질병경중등급코드로 판별해도 무방하다고 봄

# ar_rclss_cd: 발생지역구분코드

# blrs_cd: 치료행위코드                     -> 진단만받은경우(코드0,3,4,5,6,7,11,12,13,14,15) -> target=0일 확률 0퍼센트에 가까움
# show_gragh('blrs_cd')

# mdct_inu_rclss_dcd: 의료기관구분코드
show_gragh('mdct_inu_rclss_dcd')

# nur_hosp_yn: 요양병원여부
# show_gragh('nur_hosp_yn')

# optt_nbtm_s: 통원횟수                     -> 의미없음
# show_gragh('optt_nbtm_s')

# bilg_isamt_s: 청구보험금                  -> index로 나와있어서 해석하기 힘듬. 단 index>2.5면 target=0일 확룰 0에 가까움
# show_gragh('bilg_isamt_s')                #-> 86.0215, 64.5161, 53.7634 이면  target=1 일 확률 100에 가까움

# hspz_dys_s: 입원일수
# show_gragh('hspz_dys_s')

# optt_blcnt_s: 통원청구건수
# show_gragh('optt_blcnt_s')

# surop_blcnt_s: 수술청구건수                -> index > 12 이면 target=2일 확률 100에 가까움 index > 5 이면 target=0일 확률 0에 가까움
# show_gragh('surop_blcnt_s')

# hspz_blcnt_s: 입원청구건수                  -> index > 6.7 이면 targe=0일 확률 0에 가까움
# show_gragh('hspz_blcnt_s')

# hsp_avg_hspz_bilg_isamt_s: 병원별평균입원청구보험금
# show_gragh('hsp_avg_hspz_bilg_isamt_s')

# hsp_avg_optt_bilg_isamt_s: 병원별평균통원청구보험금
# show_gragh('hsp_avg_optt_bilg_isamt_s')
# hsp_avg_surop_bilg_isamt_s: 병원별평균수술청구보험금

# hsp_avg_diag_bilg_isamt_s: 병원별평균진단청구보험금


pd.set_option('display.max_rows', 500)
# # 'hsp_avg_hspz_bilg_isamt_s',
# # 'hsp_avg_optt_bilg_isamt_s', 'hsp_avg_surop_bilg_isamt_s',
# # 'hsp_avg_diag_bilg_isamt_s'
#
# data = data.head(1000)
# data1 = data[['optt_nbtm_s', 'hspz_dys_s', 'bilg_isamt_s']]
#
# print(data.head(500))
# # 1. 텐서 생성
# K = 3                    # 군집개수(k 설정)
# g = tf.Graph()
# with g.as_default():
#     vectors = tf.constant(data1)
#     centroides = tf.Variable(tf.slice(tf.random.shuffle(vectors),[0, 0], [K, -1]))
#
#     # 3. 중심과의 거리 계산
#     expanded_vectors = tf.compat.v1.expand_dims(vectors, 0)   # 3차원으로 만들어 뺄셈을 하게 해준다
#
#     expanded_centroides = tf.compat.v1.expand_dims(centroides, 1)
#
#     # 3.1 유클리드 제곱거리를 사용하는 할당단계의 알고리즘
#     diff = tf.compat.v1.subtract(expanded_vectors, expanded_centroides) # 중심 - 각 x, y 값을 뺀 것
#
#     square_diff = tf.compat.v1.square(diff)
#     distance = tf.compat.v1.reduce_sum(square_diff, 2)
#     assignments = tf.compat.v1.argmin(distance, 0)                 # 거리의 합이 가장 작은 값의 인덱스(0차원)
#
#
#
#     means = tf.concat([tf.compat.v1.reduce_mean(tf.gather(vectors,
#                                                           tf.compat.v1.reshape(tf.where(tf.equal(assignments, cluster)),
#                                                                                [1, -1])), reduction_indices=[1])
#                        for cluster in range(K)], axis=0)
#
#     # # 3.3 그래프 실행
#     # # 루프를 구성하고 중심을 means 텐서의 새 값으로 업데이트함
#     # 1) means 텐서의 값을 centroids에 할당하는 연산을 작성해야함
#
#     update_centroids = tf.compat.v1.assign(centroides, means)
#     init_op = tf.compat.v1.initialize_all_variables()
#     sess = tf.compat.v1.Session(graph=g)
#     sess.run(init_op)
#
#     for step in range(1000):
#         _, centroid_values, assignment_values = sess.run([update_centroids,
#                                                           centroides,
#                                                           assignments])
#
#     data = [int(value) for value in assignment_values]
#     df_std = StandardScaler().fit_transform(data1)
#
#     pca = decomposition.PCA(n_components=3)
#     sklean_pca_df = pca.fit_transform(df_std)
#     sklean_result = pd.DataFrame(sklean_pca_df, columns=['x', 'y', 'z'])
#     data = pd.DataFrame(data, columns=['cluster'])
#     df_final = pd.concat([sklean_result, data], axis=1)
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(df_final['x'], df_final['y'] , df_final['z']
#                          , c=df_final['cluster'], s=30, edgecolors='white')
#     legend1 = ax.legend(*scatter.legend_elements(),
#                         loc="lower left", title="groups")
#     ax.add_artist(legend1)
#     plt.show()
#
#
#     def find_row_satisfied_index(index_list, dataFrame):
#         return pd.DataFrame([dataFrame.loc[i].to_dict() for i in index_list])
#
#     def get_cluster_index_by_num(dataFrame, num):
#         return dataFrame[dataFrame['cluster'] == num].index.to_list()
#
#     group1_index = get_cluster_index_by_num(df_final,0)
#     df_group1 = find_row_satisfied_index(group1_index, data1)
#
#     group2_index = get_cluster_index_by_num(df_final, 1)
#     df_group2 = find_row_satisfied_index(group2_index, data1)
#
#
#     print(group1_index)
#     print(group2_index)