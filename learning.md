Notes
-----


Numpy - library for handing and manipulating arrays. 2d, splicing, operations.

Pandas - Built on top of numpy. Fast analysis and data cleaning and prep. Excel/R for python?

#Series
- axis labels
pd.Series(data = my_data, index=labels) - where labels is also a list. 
can pass in a list, numpy array, or dictionary

#Dataframes
pd.DataFrame(data, index, columns)

accessing dataframes. 
df['W']
df.W - confusing as column can override method

df[['W', 'Z']]

df['new_col'] = df['W'] + df['Z']

axis = 0 - rows
axis = 1 - columns

df.drop('W', axis=1, inplace=True)


columns - df['W']
rows - df.loc['A'] 

df.iloc[2]

df.loc['B', 'Y']
df.loc[['A','B'],['W', 'Y']]

sum(booleans) - only adds true values


dataframe.info attribute and .info() methods exist


## conditional selection

df > 0 - returns dataframe of boolean vals

df[df>0] - returns df where values are greater than zero

df[df['W'] > 0 ] - doesn't return Nan where columns are false

Boolean operation against series
use & instead of and

df[(df['W']>0) & (df['Y']<0)]


reset index 
df.reset_index() - removes row names, old index becomes a column of df

df.set_index('col_name')



## index hierarchy 

multi index

list(zip(list1, list2)) - list of tuples from lists

pd.MultiIndex.from_tupes()

getting from multiindex

df.loc['G1'].loc[1]

df.index.names = ['Groups', 'Num']

df.xs (cross-section) - used on multi level index

df.xs(subIndex, level='Num')


## missing data
drop or replace Nan with some value

df.dropna() - drops rows where Nan occurs (axis = 1 for cols)
threshold 

df.fillna(value=something_or_func)

## Group by

var = df.groupby('column')
var.mean()
var.sum()
count()
max()
min()
df.groupby('col').describe().transpose('col')




## concatenation, merge, joining

pd.concat([df1, df2, df3])
- can also join on axis

pd.merge()

left.join(right) - join on index instead of column.

## Operations

df['col'].unique()
df['col'].nunique()
df['col'].value_counts()

ecom['AM or PM'].value_counts() - counts occurrences of different values in that column


df['col1'].apply(my_func)

df['col1'].apply(lambda x: x &ast 2)

df.columns
df.index 
df.sort_values()

df.isnull()

## Data input and output

- Libraries to install
sqlalchemy
lxml
html5lib
BeautifulSoup4
xlrd

pd.read_csv('file_in_dir.csv')

df.to_csv('another_file.csv')

from sqlalchemy import create_engine

engine = create_engine('sqlite:///:memory;')


#Matplotlib

www.matplotlib.org



import matplotlib.pyplot as plt

%matplotlib inline 
plt.show() - if you're not using jupyter notebook

##Functional 
plt.plot(x,y)

plt.xlabel | ylabel | title 

plt.subplot() - multiple plots on same canvas

##Object-oriented approach 

fig = plt.figure() - blank canvas
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) - relative location on blank canvas: left, bottom, width, height
axes.plot(x,y)


fix, axes = plt.subplots(nrows=1, ncols=2) - 2 subplots on one canvas

axes - list of plots

plt.tight_layout() - prevents overlapping on plots


fig.savefig('mypic.png')


ax.plot(x,y, labels='')

ax.legend(loc=1)


##Colours

ax.plot(x,y, color='color', linewidth=2, alpha=0.5, linestyle='--')
 RGB hex codes #FFAA11



##Markers
ax.plot(x,y, marker='o', markersize=10)

ax.set_xlim([0,1]) - limit which portion of graph is shown


## Special plot types
Probably use Seaborn (better suited?)



#Seaborn

Statistical visualisation library
http://seaborn.pydata.org/index.html


## Distribution Plots

import seaboarn as sns
%matplotlib inline

tips = sns.load_dataset('tips')

-- histogram
sns.distplot(tips['total_bill'])

-- univariant
sns.displot(tips['total_bill'], kde=False, bins=40) - bins - how detail the histogram should be. 

sns.kdeplot() - does just kde plot, and skips histogram

-- bi variant
sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
 --  default kind = scatter, hex, reg, kde


sns.pairplot(tips) - draws graphs against all numerical data

sns.pairplot(tips, hue='sex', palette='coolwarm')
-- hue and plaette allow colorisation of category data (non-numerical)

sns.rugplot(tips['total_bill'])

kde = kernel density estimation



## Categorical plots

- non numerical 

sns.barplot(x='sex', y='total_bill', data=tips)
sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
-- std = standard deviation from numpy

sns.countplot(x='sex', data=tips) 
-- just counts per category

sns.boxplot(x='day', y='total_bill', tips, hue='smoker')
-- quartiles

sns.violinplot(x='day', y='total_bill', tips)
-- plots box plot data plus kde plot sideways 


sns.violinplot(x='day', y='total_bill', tips, hue='sex', split=True)



sns.stripplot(x='day', y='total_bill', tips)
sns.stripplot(x='day', y='total_bill', tips, jitter=True)
-- shows density a bit better
-- can add hue and split


sns.swarmplot(x='day', y='total_bill', data= 'tips')
- combo of violin and striplot

sns.factorplot(x='day', y='total_bill', data=tips, kind='bar')
-- generalisaiton of other plots
-- kind = bar, violin, scatter etc.

## Matrix plots

-- heat maps

Data should be in a matrix form - need variables on columns and rows

tc = tips.corr() - matrix form of data
sns.heatmap(tc)
sns.heatmap(tc, annot=True, cmap='coolwarm')


fp = flights.pivot_table(index='month', columns='year', values='passengers')

sns.heatmap(fp)
sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=3)


-- Cluster maps

sns.clustermap(hp)
sns.clustermap(hp, cmap='coolwarm', standar_scale=1)
-- clusters rows and columns based on similarity
-- columns and rows can be out of order, similar placed next to each other.
-- can normalise with standard_scale
