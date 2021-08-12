import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas import DataFrame

df_train: DataFrame = pd.read_csv("input/train.csv")
df_test: DataFrame = pd.read_csv("input/test.csv")

df_train.head()

#Category Name
def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))

main_categories = [c for c in df_train['category_main'].unique() if type(c)==str]
categories_sum=0
for c in main_categories:
    categories_sum+=100*len(df_train[df_train['category_main']==c])/len(df_train)
    print('{:25}{:3f}% of training data'.format(c, 100*len(df_train[df_train['category_main']==c])/len(df_train)))
print('nan\t\t\t {:3f}% of training data'.format(100-categories_sum))

df = df_train[df_train['price']<80]

my_plot = []
for i in main_categories:
    my_plot.append(df[df['category_main']==i]['price'])

fig, axes = plt.subplots(figsize=(20, 15))
bp = axes.boxplot(my_plot,vert=True,patch_artist=True,labels=main_categories)

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']*2
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes.yaxis.grid(True)

plt.title('BoxPlot price X Main product category', fontsize=15)
plt.xlabel('Main Category', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.xticks(fontsize=5)
plt.yticks(fontsize=15)
plt.show()
#####################################################################################

#3rd level
print('The data has {} unique 3rd level categories'.format(len(df_train['category_sub2'].unique())))
#Asc
df = df_train.groupby(['category_sub2'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=False, inplace=True)
df = df[:20]
df.sort_values(by=['mean_price'], ascending=True, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5)
plt.yticks(range(0,len(df)), df.index, fontsize=5)
plt.xticks(fontsize=15)
plt.title('ASCENDING - 3rd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=5)
plt.show()
########################################################################
#Dec
df = df_train.groupby(['category_sub2'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=True, inplace=True)
df = df[:50]
df.sort_values(by=['mean_price'], ascending=False, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5, color='r')
plt.yticks(range(0,len(df)), df.index, fontsize=5)
plt.xticks(fontsize=15)
plt.title('DESCENDING - 3rd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=5)
plt.show()
#####################################################################################
#2nd level
print('The data has {} unique 2nd level categories'.format(len(df_train['category_sub1'].unique())))
#Asc
df = df_train.groupby(['category_sub1'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=False, inplace=True)
df = df[:20]
df.sort_values(by=['mean_price'], ascending=True, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5, color='green')
plt.yticks(range(0,len(df)), df.index, fontsize=5)
plt.xticks(fontsize=15)
plt.title('ASCENDING - 2nd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=5)
plt.show()
########################################################################
#Dec
df = df_train.groupby(['category_sub1'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=True, inplace=True)
df = df[:50]
df.sort_values(by=['mean_price'], ascending=False, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', color='pink')
plt.yticks(range(0,len(df)), df.index, fontsize=5)
plt.xticks(fontsize=15)
plt.title('DESCENDING - 2nd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=5)
plt.show()
