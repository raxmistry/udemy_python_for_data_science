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


df['col1'].apply(my_func)

df['col1'].apply(lambda x: x &ast 2)

df.columns
df.index 
df.sort_values()

df.isnull()

## Data input and output

** Libraries to install
sqlalchemy
lxml
html5lib
BeautifulSoup4
xlrd

pd.read_csv('file_in_dir.csv')

df.to_csv('another_file.csv')

from sqlalchemy import create_engine

engine = create_engine('sqlite:///:memory;')


