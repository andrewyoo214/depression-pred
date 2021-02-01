#!/usr/bin/env python
# coding: utf-8

# ### HRV data preprocessing for FL research

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from PIL import Image


# In[2]:


# HRV 데이터셋 불러오기
hrv_df = pd.read_csv('E:/RESEARCH/Datasets/HRV_samsung/HRV_REV_all.csv', sep=',')
hrv_df.head()


# In[3]:


hrv_df.shape


# In[4]:


hrv_df['null1'] = 0
hrv_df['null2'] = 0
hrv_df['null3'] = 0
hrv_df['null4'] = 0
hrv_df['null5'] = 0
hrv_df['null6'] = 0
hrv_df['null7'] = 0
hrv_df['null8'] = 0
hrv_df['null9'] = 0
hrv_df['null10'] = 0


# In[5]:


hrv_df.shape


# In[6]:


#hrv data확인
hrv_df.head()


# In[7]:


## HRV 데이터셋에서 VISIT1, 즉 첫번째 방문에 대한 데이터만을 hrv_visit1에 저장
# hrv_visit1=hrv_df[hrv_df['VISIT']==1]
# hrv_visit1.head(10)


# In[8]:


#disorder값은 pixel에 넣지 않음. 
hrv_100 = hrv_df.drop(['sub','disorder','VISIT'], axis=1)
hrv_100.head()


# In[9]:


# 우리가 만들 분석모델의 target인 y는 disorder
y = hrv_df.loc[:,['disorder']]
y.head()


# In[10]:


# 환자만 따로 뽑으면
patient = hrv_df.loc[:, ['sub']]
patient.head


# ##### 이제 총 100개의 column으로 구성되었으니까 normalization 하자 (0~1 사이 값으로 범위변환)

# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
hrv_100[:] = scaler.fit_transform(hrv_100[:])


# ##### 만약에 평균0, 표준편차 1의 표준정규분포를 따르도록 변환하려면 아래와 같이 실행 (근데 이건 gender나 age변수에 대해서는 안좋음)

# In[12]:


# from sklearn.preprocessing import StandardScaler
# stscaler = StandardScaler()
# hrv_100[:] = stscaler.fit_transform(hrv_100[:])


# In[13]:


#normalize 제대로 되어 있는지 확인
hrv_100.head()


# ##### 여기서부터는 hrv_100 데이터를 쓰도록 하자.
# - sub, disorder, visit 변수를 제거하였음(90 rows가 되었음)
# - 10 x 10 형태를 맞춰주기 위해서 null1~null10 rows 추가
# - 각 row별로 normalize해서 0~1사이로 값을 맞춰주었음

# In[14]:


hrv_100.shape


# In[31]:


#값이 어떻게 표기되는지 x0를 통해서 확인해보자
x0=hrv_100.loc[479].values


# In[32]:


x0


# In[33]:


# 10 x 10으로 reshape
x0=x0.reshape(10,10)


# In[34]:


# heatmap으로 체크해봄. 근데 heatmap보다 그냥 plt.imsho()써서 gray scale로 보는게 가독성 있을듯
# x0 = sns.heatmap(x0)


# In[35]:


#gray scale로 확인해보고
plt.imshow(x0, cmap='gray')


# In[17]:


hrv_100.loc[0] #확인했으니까 이제 우리 데이터를 가지고


# In[23]:


hrv_100.shape


# In[30]:


hrv_100_arr = hrv_100.values
hrv_100_arr.shape


# In[31]:


# array 형태로 479개 행에 대해서 10 x 10 reshape
hrv_array = hrv_100_arr.reshape(480,10,10)


# In[32]:


hrv_array.shape


# In[275]:


# data = pd.concat([hrv_100,hrv_target], axis=1)


# In[34]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/y.npy', y)


# In[35]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/x.npy', hrv_array)


# In[37]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/patient.npy', patient)


# In[ ]:





# In[38]:


xdata = np.load('E:/RESEARCH/Datasets/HRV_samsung/x.npy')


# In[39]:


xdata


# In[ ]:





# In[ ]:





# In[289]:


hrv_target.loc[hrv_target["disorder"]==3, "disorder"] = 0


# In[280]:


x = hrv_array.reshape(-1, 10*10) #1차원 배열로 바꾸어주고
x.shape


# In[290]:


import keras
y = keras.utils.to_categorical(hrv_target, 3)


# In[298]:


y.shape


# In[ ]:





# In[ ]:




