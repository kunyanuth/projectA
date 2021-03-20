import numpy as np
a1 = [6, 4, 5, 10]
a2 = [8, 5, 3, 3]
a3 = [5, 4, 8, 4]
a4 = [4, 11, 7, 13]
a5 = [5, 8, 7, 6]
a6 = [7, 3, 5, 9]
dice = np.array([a1, a2, a3, a4, a5, a6])

from scipy import stats
stats.chi2_contingency(dice)
#https://www.mathsisfun.com/data/chi-square-test.html



dice2 = np.copy(dice)
chi2_stat, p_val, dof, ex = stats.chi2_contingency(dice2)

print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)

#Running a Large Sample to Get Expected Population Distribution
r1 = np.random.randint(1,7,1000)
r2 = np.random.randint(1,7,1000)
r3 = np.random.randint(1,7,1000)
r4 = np.random.randint(1,7,1000)
r5 = np.random.randint(1,7,1000)
print(r5)

#ตัวอย่างการใช้ np.unique
import numpy as np 
a = np.array([5,2,6,2,7,5,6,8,2,9]) 

print ('First array:' )
print (a) 
print ('\n')  

print ('Unique values of first array:' )
u = np.unique(a) 
print (u) 
print ('\n')  

print ('Return the count of repetitions of unique elements:' )
u,indices = np.unique(a,return_counts = True) 
print (u)
print (indices)

#Return the count of repetitions of unique elements 
unique, counts1 = np.unique(r1, return_counts=True)
unique, counts2 = np.unique(r2, return_counts=True)
unique, counts3 = np.unique(r3, return_counts=True)
unique, counts4 = np.unique(r4, return_counts=True)
unique, counts5 = np.unique(r5, return_counts=True)

dice3 = np.array([counts1, counts2, counts3, counts4, counts5])

chi2_stat, p_val, dof, ex = stats.chi2_contingency(dice3)

print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)

my_rolls_expected = [46.5, 46.5, 46.5, 46.5, 46.5, 46.5]
my_rolls_actual =  [59, 63, 37, 38, 32, 50]
stats.chisquare(my_rolls_actual, my_rolls_expected)


import numpy as np
from scipy.stats import chi2_contingency


table = np.array([[90, 60, 104, 95],
                  [30, 50,  51, 20],
                  [30, 40,  45, 35]])

chi2, p, dof, expected = chi2_contingency(table)

print(f"chi2 statistic:     {chi2:.5g}")
print(f"p-value:            {p:.5g}")
print(f"degrees of freedom: {dof}")
print("expected frequencies:")
print(expected)

from google.colab import files

uploaded = files.upload()

#https://pythonfordatascienceorg.wordpress.com/chi-square-python/
import pandas as pd
from scipy import stats

df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

"""The data set is from the 2016 OSMI Mental Health in Tech Survey which aims to measure attitudes towards mental health in the tech workplace, and examine the frequency of mental health disorders among tech workers.

For this example, we will test if there is an association between willingness to discuss a mental health issues with a direct supervisor and currently having a mental health disorder. 

In the data set, these are variables “Would you have been willing to discuss a mental health issue with your direct supervisor(s)?” and “Do you currently have a mental health disorder?” respectively. Let’s take a look at the data!
"""

df.shape

df.info()

df['Do you currently have a mental health disorder?'].value_counts()

df['Do you currently have a mental health disorder?'].count()

df['Would you have been willing to discuss a mental health issue with your direct supervisor(s)?'].value_counts()

"""For the variable “Do you currently have a mental health disorder?”, we are going to drop the responses of “Maybe” since we are only interested in if people know they do or do not have a mental health disorder."""

def drop_maybe(series):
    if series.lower() == 'yes' or series.lower() == 'no':
        return series
    else:
        return

df['current_mental_disorder'] = df['Do you currently have a mental health disorder?'].apply(drop_maybe)
df['current_mental_disorder'].value_counts()

df['willing_discuss_mh_supervisor'] = df['Would you have been willing to discuss a mental health issue with your direct supervisor(s)?']
df['willing_discuss_mh_supervisor'].value_counts()

pd.crosstab(df['willing_discuss_mh_supervisor'], df['current_mental_disorder'])

crosstab = pd.crosstab(df['willing_discuss_mh_supervisor'], df['current_mental_disorder'])
crosstab

stats.chi2_contingency(crosstab)

tshirts = pd.DataFrame(
    [
        [48,22,33,47],
        [35,36,42,27]
    ],
    index=["Male","Female"],
    columns=["Balck","White","Red","Blue"])
tshirts

"""![labchisquare2.png](data:image/png;base64,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)"""

tshirts.columns

chi2_stat, p_val, dof, ex = stats.chi2_contingency(tshirts)

print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)

