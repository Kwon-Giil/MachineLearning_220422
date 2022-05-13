# -*- coding: utf-8 -*-
"""
Created on 2021

@author: Administrator
"""

#%%

# =============================================================================
# =============================================================================
# # 문제 01 유형(DataSet_01.csv 이용)
#
# 구분자 : comma(“,”), 4,572 Rows, 5 Columns, UTF-8 인코딩
# 
# 글로벌 전자제품 제조회사에서 효과적인 마케팅 방법을 찾기
# 위해서 채널별 마케팅 예산과 매출금액과의 관계를 분석하고자
# 한다.
# 컬 럼 / 정 의  /   Type
# TV   /     TV 마케팅 예산 (억원)  /   Double
# Radio / 라디오 마케팅 예산 (억원)  /   Double
# Social_Media / 소셜미디어 마케팅 예산 (억원)  / Double
# Influencer / 인플루언서 마케팅
# (인플루언서의 영향력 크기에 따라 Mega / Macro / Micro / 
# Nano) / String

# SALES / 매출액 / Double
# =============================================================================
# =============================================================================

import pandas as pd
import numpy as np

df = pd.read_csv('Dataset_01.csv')

df.info()

#%%

# =============================================================================
# 1. 데이터 세트 내에 총 결측값의 개수는 몇 개인가? (답안 예시) 23
# =============================================================================

df.isna().sum().sum()
print(df.isna().sum().sum()) # 문제의 정답

tmp_df = df.copy()
tmp_df['TV'] = tmp_df['TV'].fillna(70)
tmp_df['TV'][df['TV'].isna()]
#%%

# =============================================================================
# 2. TV, Radio, Social Media 등 세 가지 다른 마케팅 채널의 예산과 매출액과의 상관분석을
# 통하여 각 채널이 매출에 어느 정도 연관이 있는지 알아보고자 한다. 
# - 매출액과 가장 강한 상관관계를 가지고 있는 채널의 상관계수를 소수점 5번째
# 자리에서 반올림하여 소수점 넷째 자리까지 기술하시오. (답안 예시) 0.1234
# =============================================================================

q2=df.corr().drop('Sales')['Sales'].abs() # Y변수 열을 추출
print(round(q2.max(), 4)) # 문제 정답

# X 변수 리스트를 추출
(q2.max()) # 최대값
q2.argmax() # 최대값이 있는 위치 번호
q2.idxmax() # 최대값이 있는 인덱스 번호
q2.nlargest(2) # 상위 2개의 변수명, 값
q2[q2 >= 0.6].index

q2.min()
q2.argmin()
q2.idxmin()
q2.nsmallest(2)


np.floor(1234.5678*100)/100 #내림
np.ceil(1234.5678*100)/100 #올림

import scipy.stats as st
tmp = df.dropna()
#st.pearsonr(df['TV'], df['Sales']) 결측지 때문에 사용불가
print(st.pearsonr(tmp['TV'], tmp['Sales'])[0])

#%%

# =============================================================================
# 3. 매출액을 종속변수, TV, Radio, Social Media의 예산을 독립변수로 하여 회귀분석을
# 수행하였을 때, 세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로
# 기술하시오. 
# - 분석 시 결측치가 포함된 행은 제거한 후 진행하며, 회귀계수는 소수점 넷째 자리
# 이하는 버리고 소수점 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================
from statsmodels.formula.api import ols
from statsmodels.api import OLS, add_constant

# 변수의 수가 많을 때를 대비해서
v_lst = ['TV','Radio','Social_Media']

# 폼을 미리 만들어준 다음
form1 = 'Sales~' + '+'.join(v_lst)

#세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로 기술
q3_lm = ols(form1, df).fit()

q3_lm.summary()

q3_lm.params.index[q3_lm.pvalues < 0.05] # 영향력 있는 변수 목록 추출

np.floor(q3_lm.params.drop('Intercept').sort_values(ascending=False).values*1000)/1000
 
#정답 [ 3.562,  0.004, -0.004]

#                              OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                  Sales   R-squared:                       0.999
# Model:                            OLS   Adj. R-squared:                  0.999
# Method:                 Least Squares   F-statistic:                 1.505e+06
# Date:                Fri, 13 May 2022   Prob (F-statistic):               0.00
# Time:                        14:45:54   Log-Likelihood:                -11366.
# No. Observations:                4546   AIC:                         2.274e+04
# Df Residuals:                    4542   BIC:                         2.277e+04
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ================================================================================
#                    coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------
# Intercept       -0.1340      0.103     -1.303      0.193      -0.336       0.068
# TV               3.5626      0.003   1051.118      0.000       3.556       3.569
# Radio           -0.0040      0.010     -0.406      0.685      -0.023       0.015
# Social_Media     0.0050      0.025      0.199      0.842      -0.044       0.054
# ==============================================================================
# Omnibus:                        0.056   Durbin-Watson:                   1.998
# Prob(Omnibus):                  0.972   Jarque-Bera (JB):                0.034
# Skew:                          -0.001   Prob(JB):                        0.983
# Kurtosis:                       3.013   Cond. No.                         149.
# ==============================================================================

#%%

# =============================================================================
# =============================================================================
# # 문제 02 유형(DataSet_02.csv 이용)
# 구분자 : comma(“,”), 200 Rows, 6 Columns, UTF-8 인코딩

# 환자의 상태와 그에 따라 처방된 약에 대한 정보를 분석하고자한다
# 
# 컬 럼 / 정 의  / Type
# Age  / 연령 / Integer
# Sex / 성별 / String
# BP / 혈압 레벨 / String
# Cholesterol / 콜레스테롤 레벨 /  String
# Na_to_k / 혈액 내 칼륨에 대비한 나트륨 비율 / Double
# Drug / Drug Type / String
# =============================================================================
# =============================================================================

import pandas as pd
import numpy as np

df1 = pd.read_csv('Dataset_02.csv')
df1.info()


#%%

# =============================================================================
# 1.해당 데이터에 대한 EDA를 수행하고, 여성으로 혈압이 High, Cholesterol이 Normal인
# 환자의 전체에 대비한 비율이 얼마인지 소수점 네 번째 자리에서 반올림하여 소수점 셋째
# 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================

df1.columns

var_list = ['Sex', 'BP', 'Cholesterol']
q1= df1[var_list].value_counts(normalize=True)

print(q1[('F', 'HIGH',  'NORMAL')]) # 정답

tab1 = pd.crosstab(index=[df1['Sex'], df1['BP']],columns=df1['Cholesterol'], margins =True)
tab2 = pd.crosstab(index=[df1['Sex'], df1['BP']],columns=df1['Cholesterol'], 
                   margins =True, normalize=True)

pd.concat([tab1, tab2], axis=1)


#%%

# =============================================================================
# 2. Age, Sex, BP, Cholesterol 및 Na_to_k 값이 Drug 타입에 영향을 미치는지 확인하기
# 위하여 아래와 같이 데이터를 변환하고 분석을 수행하시오. 
# - Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은
# ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. 
# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 
# 초과는 ‘Lv4’로 변환하시오.
# - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을
# 수행하시오.
# - 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수
# 가운데 가장 큰 p-value를 찾아 소수점 여섯 번째 자리 이하는 버리고 소수점 다섯
# 번째 자리까지 기술하시오.
# (답안 예시) 3, 1.23456
# =============================================================================

# 카이스퀘어 검정
# 1. 적합성 (범주형 변수: 1개)
# 2. 독립성 (범주형 변수: 2개)
# 3. 동질성 (범주형 변수: 2개)

# 가설: 사전에 분포를 알고 있다.
# 귀무가설 (h0): 두 변수는 독립이다 --> 모집단
# 대립가설 (h1): 두 변수는 독립이 아니다 (즉, 상관 관계가 있다.) --> 표본

# 빈도
tab1=pd.crosstab(index=df1['Sex'], columns=df1['Drug'])

# 카이스퀘어 검정 진행
from scipy.stats import chi2_contingency

chi2_out = chi2_contingency(tab1)
print(chi2_out[1]) #pvalue = 0.7138369773987128 (성별과 약물과는 연관성 없음)

# 성별과 약은 독립이다.

# 파생 변수 만들기: Age_gr, Na_K_gr
q2 = df1.copy()

# np.where(조건) --> True인 위치 번호
# np.where(조건, 참일 때, 거짓일 때)
df1['Age_gr'] = np.where(q2['Age']<20, 10, 
                        np.where(q2['Age']<30, 20,
                                 np.where(q2['Age']<40, 30,
                                          np.where(q2['Age']<50, 40,
                                                   np.where(q2['Age']<60, 50,60)))))

df1['Na_K_gr'] = np.where(q2['Na_to_K']<=10, 'Lv1',
                         np.where(q2['Na_to_K']<=20, 'Lv2',
                                  np.where(q2['Na_to_K']<=30, 'Lv3', 'Lv4')))


# - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을
# 수행하시오.
# - 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수
# 가운데 가장 큰 p-value를 찾아 소수점 여섯 번째 자리 이하는 버리고 소수점 다섯
# 번째 자리까지 기술
var_lst = ['Sex', 'BP', 'Cholesterol', 'Age_gr', 'Na_K_gr']

from scipy.stats import chi2_contingency

# Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정
q2_out=[]
for i in var_lst:

    tab2=pd.crosstab(index=df1[i], columns=df1['Drug'])
    
    chi2_out1 = chi2_contingency(tab2)
    chi2_out1[1]
    q2_out.append([i,chi2_out1[1]])
    
# 데이터 변환    
q2_out = pd.DataFrame(q2_out, columns=['var','pvalue'])
q2_out

#	var	pvalue
#0	Sex	0.7138369773987128
#1	BP	5.0417334144665895e-27
#2	Cholesterol	0.0005962588389856497
#3	Age_gr	0.0007010113024729462
#4	Na_K_gr	1.1254641594413981e-14

 # 연관성 있는 변수 개수, 가장 큰 pvalue 
print(len(q2_out[q2_out.pvalue < 0.05]['var']),
      round(q2_out[q2_out.pvalue < 0.05]['pvalue'].max(),5))

 
#%%

# =============================================================================
# 3.Sex, BP, Cholesterol 등 세 개의 변수를 다음과 같이 변환하고 의사결정나무를 이용한
# 분석을 수행하시오.
# - Sex는 M을 0, F를 1로 변환하여 Sex_cd 변수 생성
# - BP는 LOW는 0, NORMAL은 1 그리고 HIGH는 2로 변환하여 BP_cd 변수 생성
# - Cholesterol은 NORMAL은 0, HIGH는 1로 변환하여 Ch_cd 생성
# - Age, Na_to_k, Sex_cd, BP_cd, Ch_cd를 Feature로, Drug을 Label로 하여 의사결정나무를
# 수행하고 Root Node의 split feature와 split value를 기술하시오. 
# 이 때 split value는 소수점 셋째 자리까지 반올림하여 기술하시오. (답안 예시) Age, 
# 12.345
# =============================================================================
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_text

# 결정트리를 만들기 전에 데이터 전처리
df1['Sex_cd'] = np.where(df1['Sex']=='M',0 , 1)
df1['BP_cd'] = np.where(df1['BP']=='LOW',0 ,
                        np.where(df1['BP']=='NORMAL',1, 2))
df1['Ch_cd'] = np.where(df1['Cholesterol']=='NORMAL',0 , 1)

var_list = ['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']

# 결정트리를 구성
dt = DecisionTreeClassifier().fit(df1[var_list], df1.Drug)

dir(dt)

pd.Series(dt.feature_importances_, index=var_list)

# 도식화
plot_tree(dt, feature_names=var_list, class_names=df1.Drug.unique(),
          max_depth = 2,fontsize = 7)

print(export_text(dt, feature_names = var_list, decimals = 3))

# Na_to_K, 14.829

#%%

# =============================================================================
# =============================================================================
# # 문제 03 유형(DataSet_03.csv 이용)
# 
# 구분자 : comma(“,”), 5,001 Rows, 8 Columns, UTF-8 인코딩
# 안경 체인을 운영하고 있는 한 회사에서 고객 사진을 바탕으로 안경의 사이즈를
# 맞춤 제작하는 비즈니스를 기획하고 있다. 우선 데이터만으로 고객의 성별을
# 파악하는 것이 가능할 지를 연구하고자 한다.
#
# 컬 럼 / 정 의 / Type
# long_hair / 머리카락 길이 (0 – 길지 않은 경우 / 1 – 긴
# 경우) / Integer
# forehead_width_cm / 이마의 폭 (cm) / Double
# forehead_height_cm / 이마의 높이 (cm) / Double
# nose_wide / 코의 넓이 (0 – 넓지 않은 경우 / 1 – 넓은 경우) / Integer
# nose_long / 코의 길이 (0 – 길지 않은 경우 / 1 – 긴 경우) / Integer
# lips_thin / 입술이 얇은지 여부 0 – 얇지 않은 경우 / 1 –
# 얇은 경우) / Integer
# distance_nose_to_lip_long / 인중의 길이(0 – 인중이 짧은 경우 / 1 – 인중이
# 긴 경우) / Integer
# gender / 성별 (Female / Male) / String
# =============================================================================
# =============================================================================



#%%

# =============================================================================
# 1.이마의 폭(forehead_width_cm)과 높이(forehead_height_cm) 사이의
# 비율(forehead_ratio)에 대해서 평균으로부터 3 표준편차 밖의 경우를 이상치로
# 정의할 때, 이상치에 해당하는 데이터는 몇 개인가? (답안 예시) 10
# =============================================================================










#%%

# =============================================================================
# 2.성별에 따라 forehead_ratio 평균에 차이가 있는지 적절한 통계 검정을 수행하시오.
# - 검정은 이분산을 가정하고 수행한다.
# - 검정통계량의 추정치는 절대값을 취한 후 소수점 셋째 자리까지 반올림하여
# 기술하시오.
# - 신뢰수준 99%에서 양측 검정을 수행하고 결과는 귀무가설 기각의 경우 Y로, 그렇지
# 않을 경우 N으로 답하시오. (답안 예시) 1.234, Y
# =============================================================================












#%%

# =============================================================================
# 3.주어진 데이터를 사용하여 성별을 구분할 수 있는지 로지스틱 회귀분석을 적용하여
# 알아 보고자 한다. 
# - 데이터를 7대 3으로 나누어 각각 Train과 Test set로 사용한다. 이 때 seed는 123으로
# 한다.
# - 원 데이터에 있는 7개의 변수만 Feature로 사용하고 gender를 label로 사용한다.
# (forehead_ratio는 사용하지 않음)
# - 로지스틱 회귀분석 예측 함수와 Test dataset를 사용하여 예측을 수행하고 정확도를
# 평가한다. 이 때 임계값은 0.5를 사용한다. 
# - Male의 Precision 값을 소수점 둘째 자리까지 반올림하여 기술하시오. (답안 예시) 
# 0.12
# 
# 
# (참고) 
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# train_test_split 의 random_state = 123
# =============================================================================














#%%

# =============================================================================
# =============================================================================
# # 문제 04 유형(DataSet_04.csv 이용)
#
#구분자 : comma(“,”), 6,718 Rows, 4 Columns, UTF-8 인코딩

# 한국인의 식생활 변화가 건강에 미치는 영향을 분석하기에 앞서 육류
# 소비량에 대한 분석을 하려고 한다. 확보한 데이터는 세계 각국의 1인당
# 육류 소비량 데이터로 아래와 같은 내용을 담고 있다.

# 컬 럼 / 정 의 / Type
# LOCATION / 국가명 / String
# SUBJECT / 육류 종류 (BEEF / PIG / POULTRY / SHEEP) / String
# TIME / 연도 (1990 ~ 2026) / Integer
# Value / 1인당 육류 소비량 (KG) / Double
# =============================================================================
# =============================================================================

# (참고)
# #1
import pandas as pd
import numpy as np
# #2
from scipy.stats import ttest_rel
# #3


df_meat = pd.read_csv('Dataset_04.csv')
df_meat

#%%

# =============================================================================
# 1.한국인의 1인당 육류 소비량이 해가 갈수록 증가하는 것으로 보여 상관분석을 통하여
# 확인하려고 한다. 
# - 데이터 파일로부터 한국 데이터만 추출한다. 한국은 KOR로 표기되어 있다.
# - 년도별 육류 소비량 합계를 구하여 TIME과 Value간의 상관분석을 수행하고
# 상관계수를 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지만 기술하시오. 
# (답안 예시) 0.55
# =============================================================================
# 한국 데이터만 추출
df_Kmeat = df_meat[df_meat.LOCATION =='KOR']

# df_Kmeat['Time'].value_counts() # data 탐색용

print(round(df_meat.groupby('TIME')['Value'].sum().reset_index(drop = False).corr(),))


#%%

# =============================================================================
# 2. 한국 인근 국가 가운데 식생의 유사성이 상대적으로 높은 일본(JPN)과 비교하여, 연도별
# 소비량에 평균 차이가 있는지 분석하고자 한다.
# - 두 국가의 육류별 소비량을 연도기준으로 비교하는 대응표본 t 검정을 수행하시오.
# - 두 국가 간의 연도별 소비량 차이가 없는 것으로 판단할 수 있는 육류 종류를 모두
# 적으시오. (알파벳 순서) (답안 예시) BEEF, PIG, POULTRY, SHEEP
# =============================================================================

# t 검정
# (1). 일표본 t 검정 
# (2). 이표본 t 검정
# - 독립 / 대응

from scipy.stats import ttest_rel # 대응형 t검정을 수행할 때 사용

sub_lst = df_meat.SUBJECT.unique()

df_kjmeat=df_meat[df_meat.LOCATION.isin(['KOR', 'JPN'])] # 해당 데이터만 필터링
temp_data = df_meat[df_meat.SUBJECT == 'BEEF']
table = pd.pivot(temp_data, 
                 index='TIME', 
                 columns="LOCATION", 
                 values='Value').dropna()

ttest_out=ttest_rel(table['KOR'], table['JPN'])
ttest_out.pvalue # = 2.6431630595494026e-08

# h0(귀무가설): 두집단 평균은 같다 muA == muB, muA - muB = 0
# h1(대립가설): 두집단의 평균은 같지 않다. muA != muB, muA - muB != 0

q2_res=[]

for i in sub_lst:
    df_kjmeat=df_meat[df_meat.LOCATION.isin(['KOR', 'JPN'])] # 해당 데이터만 필터링
    temp_data = df_meat[df_meat.SUBJECT == i]
    table = pd.pivot(temp_data, 
                     index='TIME', 
                     columns="LOCATION", 
                     values='Value').dropna()
    
    ttest_out=ttest_rel(table['KOR'], table['JPN'])
    pvalue=ttest_out.pvalue # = 2.6431630595494026e-08
    q2_res.append([i, pvalue])
    
q2_res = pd.DataFrame(q2_res, columns =['sub', 'pvalue'])
q2_res[q2_res.pvalue>=0.05]['sub']

# 정답: POULTRY

from scipy.stats import ttest_ind, bartlett, levene

temp_data = df_meat[df_meat.SUBJECT == 'BEEF']
tab1 = pd.pivot(temp_data, 
                 index='TIME', 
                 columns="LOCATION", 
                 values='Value')

bartlett(tab1['KOR'].dropna(), tab1['JPN'].dropna())

#h0: 등분산 / h1: 이분산 (등분산 아니다)

tt_out=ttest_ind(table['KOR'], table['JPN'], equal_var = False)
pvalue1 = tt_out.pvalue # = 2.6431630595494026e-08 
pvalue1
#%%

# =============================================================================
# 3.(한국만 포함한 데이터에서) Time을 독립변수로, Value를 종속변수로 하여 육류
# 종류(SUBJECT) 별로 회귀분석을 수행하였을 때, 가장 높은 결정계수를 가진 모델의
# 학습오차 중 MAPE를 반올림하여 소수점 둘째 자리까지 기술하시오. (답안 예시) 21.12
# (MAPE : Mean Absolute Percentage Error, 평균 절대 백분율 오차)
# (MAPE = Σ ( | y - y ̂ | / y ) * 100/n ))
# 
# =============================================================================

from sklearn.linear_model import LinearRegression #상수항 선택 가능
from statsmodels.formula.api import ols #상수항 선택 가능
from statsmodels.api import OLS, add_constant # 상수항 텀 추가 후 회귀 진행

# 1. 한국만 포함한 데이터에서
df_Kmeat = df_meat[df_meat.LOCATION =='KOR']


# 2. Time을 독립변수, Value를 종속변수로 종류별로 회귀분석 수행

# (1) LinearRegression: X는 2차식(2차원), y는 1차식(1차원)
lr1 = LinearRegression(fit_intercept = True).fit(df_Kmeat[['TIME']],df_Kmeat['Value'])

dir(lr1)


lr1.intercept_
lr1.coef_

# 결정 계수를 이용할 때
lr1.score(df_Kmeat[['TIME']],df_Kmeat['Value'])

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

pred = lr1.predict(df_Kmeat[['TIME']])
mse=mean_absolute_error(df_Kmeat['Value'], pred)

#(MAPE : Mean Absolute Percentage Error, 평균 절대 백분율 오차)
# (MAPE = Σ ( | y - y ̂ | / y ) * 100/n ))
mape = (abs(df_Kmeat["Value"] - pred) / df_Kmeat["Value"]).sum()*100/len(df_Kmeat)


# 육류 종류 별로 완성하기
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sub_lst = df_Kmeat.SUBJECT.unique()
sub_lst

# 반복을 돌려서 선형 회귀를 반복하여 육류 종류 별로 회귀 결과 도출

q3_out=[]
for i in sub_lst:
    temp = df_Kmeat[df_Kmeat.SUBJECT == i]
    
    lr2 = LinearRegression(fit_intercept = True).fit(temp[['TIME']],temp['Value'])
    r2=lr2.score(temp[['TIME']],temp['Value'])
    
    pred = lr2.predict(temp[['TIME']])
    
    #(MAPE : Mean Absolute Percentage Error, 평균 절대 백분율 오차)
    # (MAPE = Σ ( | y - y ̂ | / y ) * 100/n ))
    mape = (abs(temp["Value"] - pred) / temp["Value"]).sum()*100/len(df_Kmeat)
    q3_out.append([i,r2,mape])

# 가장 높은 결정 계수를 가진 모델의 학습 오차 중 mape 값 도출 및 반올림
    
q3_out = pd.DataFrame(q3_out, columns=['sub', 'r2', 'mape'])

round(q3_out.loc[q3_out.r2.idxmax(),'mape'],2)

## [참고]
print(df_Kmeat['TIME'].shape)
print(df_Kmeat['TIME'].ndim)
print(df_Kmeat[['TIME']].shape)
print(df_Kmeat[['TIME']].ndim)
print(df_Kmeat['TIME'].values.reshape(-1,1).shape)
print(df_Kmeat['TIME'].values.reshape(-1,1).ndim)

# [참고2] 조금 더 코드 모양을 깔끔하게 만드는 방법
for i in sub_lst:
    temp = df_Kmeat[df_Kmeat.SUBJECT == i]
    
    globals()['lr_'+i] = LinearRegression(fit_intercept = True).fit(temp[['TIME']],temp['Value'])
    
#lr_BEEF.coef_
#eval('lr_'+i)

# ols
# 문법: ols('y~x1+x2+x3-1', dataset).fit()
# lm1 = ols('y~x1+x2+x3-1', dataset) # 다른 모델에 반복해서 써야될 때
# lm2 = lm1.fit() 

lm2=ols('Value~TIME', df_Kmeat).fit()
dir(lm2)
lm2.summary()

#OLS Regression Results                            
#==============================================================================
#Dep. Variable:                  Value   R-squared:                       0.107
#Model:                            OLS   Adj. R-squared:                  0.101
#Method:                 Least Squares   F-statistic:                     17.21
#Date:                Fri, 13 May 2022   Prob (F-statistic):           5.70e-05
#Time:                        13:17:28   Log-Likelihood:                -516.62
#No. Observations:                 146   AIC:                             1037.
#Df Residuals:                     144   BIC:                             1043.
#Df Model:                           1                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                coef      std err       t        P>|t|      [0.025      0.975]
#----------
#Intercept   -537.6899    132.239     -4.066      0.000    -799.070    -276.310
#TIME           0.2731      0.066      4.148      0.000       0.143       0.403
#==============================================================================
#==============================================================================
#Omnibus:                        6.442   Durbin-Watson:                   0.043
#Prob(Omnibus):                  0.040   Jarque-Bera (JB):                3.528
#Skew:                           0.153   Prob(JB):                        0.171
#Kurtosis:                       2.303   Cond. No.                     3.83e+05
#==============================================================================

from statsmodels.stats.anova import anova_lm
anova_lm(lm2)

# F-statistic:                     17.21
# Prob (F-statistic):           5.70e-05

# h0[귀무]: B1 = B2 = B3
# h1[대립]: 적어도 i하나에 대해서 Bi! = 0
# 유의수준 0.05 하에 pvalue가 유의수준보다 작으면 귀무가설 기각
# 유의 수준 0.05하에서 pvalue가 5.70e-05보다 작으므로 귀무가설 기각


# h0:b0 == 0
# h1:b1 != 0

# h0:b0 == 0
# h1:b1 != 0

df_Kmeat[['TIME', "Value"]].plot(kind='scatter', x='TIME', y='Value')
lm3 = ols('Value~TIME+SUBJECT',df_Kmeat).fit()
lm3.summary()

#OLS Regression Results                            
#==============================================================================
#Dep. Variable:                  Value   R-squared:                       0.926
#Model:                            OLS   Adj. R-squared:                  0.924
#Method:                 Least Squares   F-statistic:                     442.6
#Date:                Fri, 13 May 2022   Prob (F-statistic):           1.02e-78
#Time:                        13:45:08   Log-Likelihood:                -334.56
#No. Observations:                 146   AIC:                             679.1
#Df Residuals:                     141   BIC:                             694.0
#Df Model:                           4                                         
#Covariance Type:            nonrobust                                         
#======================================================================================
#                         coef    std err          t      P>|t|      [0.025      0.975]
#--------------------------------------------------------------------------------------
#Intercept           -520.0777     38.422    -13.536      0.000    -596.036    -444.119
#SUBJECT[T.PIG]        13.9952      0.574     24.383      0.000      12.861      15.130
#SUBJECT[T.POULTRY]     4.3373      0.570      7.607      0.000       3.210       5.465
#SUBJECT[T.SHEEP]      -8.1322      0.570    -14.263      0.000      -9.259      -7.005
#TIME                   0.2631      0.019     13.756      0.000       0.225       0.301
#==============================================================================
#Omnibus:                        2.541   Durbin-Watson:                   0.205
#Prob(Omnibus):                  0.281   Jarque-Bera (JB):                2.528
#Skew:                          -0.314   Prob(JB):                        0.283
#Kurtosis:                       2.858   Cond. No.                     3.83e+05
#==============================================================================

tmp_df = df_Kmeat[['TIME','SUBJECT','Value']]
tmp_dum = pd.get_dummies(tmp_df, columns=['SUBJECT'],drop_first=True)
tmp_onehot = pd.get_dummies(tmp_df, columns=['SUBJECT'],drop_first=False)

pred2 = lm2.predict(df_Kmeat['TIME'])
pred3 = lm3.predict(df_Kmeat[['TIME', 'SUBJECT']])

#print(mean_sqared_error(df_Kmeat[['Value']], pred2))
print(mean_squared_error(df_Kmeat['Value'], pred2)) # 69.34278088859985
mean_squared_error(df_Kmeat['Value'], pred3) # 5.726854954500384
#%%

# =============================================================================
# =============================================================================
# # 문제 05 유형(DataSet_05.csv 이용)
#
# 구분자 : comma(“,”), 8,068 Rows, 12 Columns, UTF-8 인코딩
#
# A자동차 회사는 신규 진입하는 시장에 기존 모델을 판매하기 위한 마케팅 전략을 
# 세우려고 한다. 기존 시장과 고객 특성이 유사하다는 전제 하에 기존 고객을 세분화하여
# 각 그룹의 특징을 파악하고, 이를 이용하여 신규 진입 시장의 마케팅 계획을 
# 수립하고자 한다. 다음은 기존 시장 고객에 대한 데이터이다.
#

# 컬 럼 / 정 의 / Type
# ID / 고유 식별자 / Double
# Age / 나이 / Double
# Age_gr / 나이 그룹 (10/20/30/40/50/60/70) / Double
# Gender / 성별 (여성 : 0 / 남성 : 1) / Double
# Work_Experience / 취업 연수 (0 ~ 14) / Double
# Family_Size / 가족 규모 (1 ~ 9) / Double
# Ever_Married / 결혼 여부 (Unknown : 0 / No : 1 / Yes : 2) / Double
# Graduated / 재학 중인지 여부 / Double
# Profession / 직업 (Unknown : 0 / Artist ~ Marketing 등 9개) / Double
# Spending_Score / 소비 점수 (Average : 0 / High : 1 / Low : 2) / Double
# Var_1 / 내용이 알려지지 않은 고객 분류 코드 (0 ~ 7) / Double
# Segmentation / 고객 세분화 결과 (A ~ D) / String
# =============================================================================
# =============================================================================


#(참고)
#1
# import pandas as pd
# #2
# from scipy.stats import chi2_contingency
# #3
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
# import pydot


#%%

# =============================================================================
# 1.위의 표에 표시된 데이터 타입에 맞도록 전처리를 수행하였을 때, 데이터 파일 내에
# 존재하는 결측값은 모두 몇 개인가? 숫자형 데이터와 문자열 데이터의 결측값을
# 모두 더하여 답하시오.
# (String 타입 변수의 경우 White Space(Blank)를 결측으로 처리한다) (답안 예시) 123
# =============================================================================






#%%

# =============================================================================
# 2.이어지는 분석을 위해 결측값을 모두 삭제한다. 그리고, 성별이 세분화(Segmentation)에
# 영향을 미치는지 독립성 검정을 수행한다. 수행 결과, p-value를 반올림하여 소수점
# 넷째 자리까지 쓰고, 귀무가설을 기각하면 Y로, 기각할 수 없으면 N으로 기술하시오. 
# (답안 예시) 0.2345, N
# =============================================================================





#%%

# =============================================================================
# 3.Segmentation 값이 A 또는 D인 데이터만 사용하여 의사결정 나무 기법으로 분류
# 정확도를
# 측정해 본다. 
# - 결측치가 포함된 행은 제거한 후 진행하시오.
# - Train대 Test 7대3으로 데이터를 분리한다. (Seed = 123)
# - Train 데이터를 사용하여 의사결정나무 학습을 수행하고, Test 데이터로 평가를
# 수행한다.
# - 의사결정나무 학습 시, 다음과 같이 설정하시오:
# • Feature: Age_gr, Gender, Work_Experience, Family_Size, 
#             Ever_Married, Graduated, Spending_Score
# • Label : Segmentation
# • Parameter : Gini / Max Depth = 7 / Seed = 123
# 이 때 전체 정확도(Accuracy)를 소수점 셋째 자리 이하는 버리고 소수점 둘째자리까지
# 기술하시오.
# (답안 예시) 0.12
# =============================================================================



