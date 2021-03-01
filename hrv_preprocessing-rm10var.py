#!/usr/bin/env python
# coding: utf-8

# ### HRV data preprocessing for FL research

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from PIL import Image


# In[7]:


# HRV 데이터셋 불러오기
hrv_df = pd.read_csv('E:/RESEARCH/Datasets/HRV_samsung/HRV_REV_all.csv', sep=',')
hrv_df.head()


# In[8]:


hrv_df.shape


# In[9]:


hrv_df['null1'] = 0
# hrv_df['null2'] = 0
# hrv_df['null3'] = 0
# hrv_df['null4'] = 0
# hrv_df['null5'] = 0
# hrv_df['null6'] = 0
# hrv_df['null7'] = 0
# hrv_df['null8'] = 0
# hrv_df['null9'] = 0
# hrv_df['null10'] = 0


# In[10]:


hrv_df.shape


# In[11]:


#hrv data확인
hrv_df.head()


# In[29]:


#disorder, sub, Visit, 10개변수 값은 pixel에 넣지 않음. (총 13개 제거)
hrv_81 = hrv_df.drop(['sub','disorder','VISIT','HAMD', 'HAMA','PDSS','ASI','APPQ','PSWQ','SPI','PSS','BIS','SSI'], axis=1)
hrv_81.head()


# In[15]:


# target을 disorder로 놓으면
y = hrv_df.loc[:,['disorder']]


# In[16]:


# target을 HAMD로 놓으면
y1 = hrv_df.loc[:,['HAMD']]


# In[17]:


# target을 HAMA로 놓으면
y2 = hrv_df.loc[:,['HAMA']]


# In[18]:


# target을 PDSS로 놓으면
y3 = hrv_df.loc[:,['PDSS']]


# In[19]:


# target을 ASI로 놓으면
y4 = hrv_df.loc[:,['ASI']]


# In[20]:


# target을 APPQ로 놓으면
y5 = hrv_df.loc[:,['APPQ']]


# In[21]:


# target을 PSWQ로 놓으면
y6 = hrv_df.loc[:,['PSWQ']]


# In[22]:


# target을 SPI로 놓으면
y7 = hrv_df.loc[:,['SPI']]


# In[23]:


# target을 PSS로 놓으면
y8 = hrv_df.loc[:,['PSS']]


# In[26]:


# target을 BIS로 놓으면
y9 = hrv_df.loc[:,['BIS']]


# In[27]:


# target을 SSI로 놓으면
y10 = hrv_df.loc[:,['SSI']]


# In[28]:


# 환자만 따로 뽑으면
patient = hrv_df.loc[:, ['sub']]


# In[ ]:





# In[ ]:





# ##### 이제 총 81개의 column으로 구성되었으니까 우선 normalization 하자 (0~1 사이 값으로 범위변환)

# In[30]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
hrv_81[:] = scaler.fit_transform(hrv_81[:])


# ##### 만약에 평균0, 표준편차 1의 표준정규분포를 따르도록 변환하려면 아래와 같이 실행 (근데 이건 gender나 age변수에 대해서는 안좋음)

# In[11]:


# from sklearn.preprocessing import StandardScaler
# stscaler = StandardScaler()
# hrv_100[:] = stscaler.fit_transform(hrv_100[:])


# In[31]:


#normalize 제대로 되어 있는지 확인
hrv_81.head()


# ##### 여기서부터는 hrv_81 데이터를 쓰도록 하자.
# - sub, disorder, visit, 10개주요 변수를 제거하였음(80 rows가 되었음)
# - 9 x 9 형태를 맞춰주기 위해서 null1 rows 추가
# - 각 row별로 normalize해서 0~1사이로 값을 맞춰주었음

# In[32]:


hrv_81.shape


# In[33]:


#값이 어떻게 표기되는지 x0를 통해서 확인해보자
x0=hrv_81.loc[479].values


# In[34]:


x0


# In[35]:


# 9 x 9으로 reshape
x0=x0.reshape(9,9)


# In[36]:


# heatmap으로 체크해봄. 근데 heatmap보다 그냥 plt.imsho()써서 gray scale로 보는게 가독성 있을듯
# x0 = sns.heatmap(x0)


# In[37]:


#gray scale로 확인해보고
plt.imshow(x0, cmap='gray')


# In[38]:


hrv_81.loc[0] #확인했으니까 이제 우리 데이터를 가지고


# In[40]:


hrv_81.shape


# In[41]:


hrv_81_arr = hrv_81.values
hrv_81_arr.shape


# In[42]:


# array 형태로 480개 행에 대해서 9 x 9 reshape
hrv_array = hrv_81_arr.reshape(480,9,9)


# In[43]:


hrv_array.shape


# In[24]:


# data = pd.concat([hrv_100,hrv_target], axis=1)


# In[25]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y.npy', y)


# In[44]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y1.npy', y1)


# In[45]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y2.npy', y2)


# In[46]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y3.npy', y3)


# In[47]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y4.npy', y4)


# In[48]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y5.npy', y5)


# In[49]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y6.npy', y6)


# In[50]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y7.npy', y7)


# In[51]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y8.npy', y8)


# In[52]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y9.npy', y9)


# In[53]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/y10.npy', y10)


# In[ ]:





# In[54]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/x.npy', hrv_array)


# In[55]:


np.save('E:/RESEARCH/Datasets/HRV_samsung/pracdata/patient.npy', patient)


# In[ ]:





# In[56]:


# 확인해보자
xdata = np.load('E:/RESEARCH/Datasets/HRV_samsung/pracdata/x.npy')


# In[57]:


xdata


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




