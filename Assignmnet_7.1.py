#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


Task:
1. fitting it to the periodic function
2. plot the fit


# In[8]:


import numpy as np


# In[5]:


Temp_Max =np.array([39, 41, 43, 47, 49, 51, 45, 38, 37, 29, 27, 25])
Temp_Min =np.array([21, 23, 27, 28, 32, 35, 31, 28, 21, 19, 17, 18])


# In[7]:


Temp_Max
Temp_Min


# In[12]:


months=np.arange(12)
plt.plot(months,Temp_Max,'ro')
plt.plot(months,Temp_Min,'bo')
plt.xlabel('Month')
plt.ylabel('Min and Max Temperatures ($^\circ$C)')


# In[15]:


from scipy import optimize
def yearly_temps(times,avg,ampl,time_offset):
    return (avg+ampl*np.cos((times+time_offset)*2*np.pi/times.max()))
res_max,cov_max=optimize.curve_fit(yearly_temps,months,Temp_Max,[20,10,0])
res_min,cov_min=optimize.curve_fit(yearly_temps,months,Temp_Min,[-40,20,0])


# In[16]:


days=np.linspace(0,12,num=365)
plt.figure()
plt.plot(months,Temp_Max,'ro')
plt.plot(days,yearly_temps(days,*res_max),'r-')
plt.plot(months,Temp_Min,'bo')
plt.plot(days,yearly_temps(days,*res_min),'b-')
plt.xlabel('Month')
plt.ylabel('Temperature ($^\circ$C)')
plt.show()


# In[1]:


import pandas as pd


# In[2]:


titanic = pd.read_csv('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv')


# In[3]:


titanic.head()


# 2.Create a pie chart presenting the male/female proportion

# In[14]:


female_count = titanic.groupby(by="sex").size().female
male_count = titanic.groupby(by="sex").size().male
plt.figure(1, figsize=(6,6))
labels = ["Male", "Famale"]
sizes = [male_count, female_count]
colors = ['blue', 'red']
explode=(0, 0.05)
plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90, explode=explode)
plt.title('Sex Proportion')
#plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()



# 3.Create a scatterplot with the Fare paid and the Age, differ the plot color by gender

# In[15]:


plt.scatter(titanic['age'], titanic['fare'], alpha=0.5, c=pd.factorize(titanic['sex'])[0])


# In[ ]:




