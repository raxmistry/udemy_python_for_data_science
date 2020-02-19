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



## Grids

g = sns.PairGrid(iris) -- creates empty grids, more control.
g.map(plt.scatter)

g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.distplot, 'total_bill')
g.map(sns.scatterplot, 'total_bill', 'tip')



## Regression plots

sns.lmplot(x='total_bill', y='tip', tips, hue='sex')
sns.lmplot(x='total_bill', y='tip', tips, hue='sex', markers=['o', 'v'], scatter_kws{'s': 100})

-- scatter_kws takes in a dictionary and calls matplotlib underneath



sns.lmplot(x='total_bill', y='tip', tips, col='sex', row='time', aspect=0.6, size=8)
-- breaks it into facets



## Style and colour

sns.set_style('white') -- ticks, darkgrid, whitegrid
sns.countplot(x='sex', data=tips)

sns.despine( ) - removes top and right spines


plt.figure(figsize=(12,3))
-- matplotlib figure size and aspect will affect seaborn

sns.set_context('poster', font_scale=3) -- notebook option


other palette options on matplotlib colormaps 


# Pandas built-in data visualisation

df1 = pd.read_csv('df1')

-- 3 different ways
df1['A'].hist(bins=30)

df1['A'].plot(kind='hist', bins=30)

df1['A'].plot.hist()




df2.plot.area()
df2.plot.area(alpha)

df2.plot.bar(stacked=True)


df1.plot.line(x=df1.index, y='B', figsize=(12,3), lw=1)


-- all matplotlib args avaialble here 

df1.plot.scatter(x='A', y='B', c='C') - c color , cmap
df1.plot.scatter(x='A', y='B', s=df1['C']\*100) - size


df.plot.hexbin(x='a', x='b', gridsize=25, cmap='coolwarm')

df2['a'].plot.kde()
df2['a'].plot.density()



# Plotly and Cufflinks

pip install plotly
pip install cufflinks


cufflinks connects plotly with pandas
plotly - interactive visualisation?

from plotly import __version__
print(__version__)

Need > 1.9 required

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()

df.iplot() (instead of pandas df.plot())

df.iplot(kind='scatter', x='A', y='B', mode='markers')

df.iplot(kind='scatter', x='A', y='B', mode='markers', size=20)

df.iplot(kind='bar', x='Categories', y='B')

df.sum().iplot(kind='bar')

df.iplot(kind='box')

df3.iplot(kind='surface', colorscale='rdylbu')

df['A'].iplot(kind='hist', bins=50)
df.iplot(kind='hist', bins=50)


df[['A', 'B']].iplot(kind='spread') -- useful for financial data

df.iplot(kind='bubble', x='A', y='B', size='C')

df.scatter_matrix()

For documentation look at cufflinks github page:
https://github.com/santosjorge/cufflinks

Financial analysis = cufflinks, ta.py

# Geographical plotting

plotly going to be used
but can use matplotlib basemap extension as an alternative


## Choropleth maps

import plotly.plotly as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)

-- create a dict or hashmap of key values
data = dict( type = 'chloropeth', 
locations = ['AZ', 'CA', 'NY'],
locationmode = 'USA-states',
colorscale = 'Portland',
text = ['text 1', 'text 2', 'text 3'],
z= [1.0, 2.0, 3.0],
colorbar = {'title': 'color bar title '} )

layout = dict(geo={'scope': 'usa'})
import plotly.graph_objs as ?go

choromap = go.Figure(data = [data], layout = layout)

iplot(choromap)


Global Choropleth Plots


# Data Capstone Project



# Part 2 - Machine Learning 

## Supervised learning

algorithms trainined using labelled data
labeled examples


learns with inputs and correct outputs and modifies model.

ML Process

Data Acquisition
Data cleaning
Training data & Test data
Use training data fit and build model
Model testing
Adjust model parameters
Model deployment

Data split into 3 sets
- Training data
- Validation data
   -- to adjust model hyperparameters
- Test Data 
   -- final performance metrics 
   -- once used this data, not allowed to adjust parameters again.


# Evaluating performance - classification problems
Key metrics
- accuracy
- recall
- precision
- F1-Score

Binary classification - only 2 classes

Model evaluation 
- in real world not all correct or incorrect matches have the same weighting

- confusion matrix

Accuracy in classification problems os the number of correct predictions made by the mode divided by the total number of predictions. 
- useful when target classes are well balanced - labels equally representative
- unbalanced classes - accuracy not that useful


Recall is ability of a model to find all relevant cases withing a dataset. 
Number of true positives divided by the number of true positives plus the number of false negatives. 

Precision ability of a classification model to identify only the relevant data points. 
number of true positives divided by the number of true positives plus the number of false positives.

Often trade-off between recall and precision

While recall expresses the ability to find all relevant instaces in a dataset, precision expresses the proportion of the data points our model says were relevant that actually were relelvant. 

F1 - Score 
optimal blend of precision and recall 
harmonic mean of precision and recall taking both metrics into account 


F1 = 2\*(precision\*recall)/(precision+recall)

harmonic mean punishes extreme differences between precision and recall

Confusion matrix 
correctly classified vs incorrectly classified in form of confusion matrix

False positive - type 1 error
False Negative - type 11 error 

further metrics can be calculated off the confusion matrix

Check wikipedia

Good enough accuracy ? - depends on context



## Model performance evaluation - regression error metrics

Regression is a task when a model attempts to predict continuous values (unlike categorical values which is classification)

New metrics needed for regression 

Mean Absolute Error - differences between prediction and true value.
Does not punish large errors


Mean Squared Error - difference between true and predicted values, squared
- punishes large errors 
- squares units as well - which makes models difficult to read

Therefore

Root Mean Squared Error - square root of Mean Squared Error
(RMSE)

Compare error metric to average value of label to understand performance of metric

## Machine learning with Python

Scikit Learn package

conda install scikit-learn


has family of models

download - 
scikit-learn algorithm cheat sheet


## Intro to Linear Regression 


Just a quick note, in Late September 2016, SciKit Learn 0.18 was released and there was a slight change to the code. With 0.18 the train_test_split function is now imported from model_selection instead of cross_validation. This is already reflected in the latest notes, but the videos show the older method. You can still use the older method, you'll just get a warning message notifying you of the change. So for short, the statement:

from sklearn.cross_validation import train_test_split

has been changed to :

from sklearn.model_selection import train_test_split

The same has also happened for GridSearchCV (we'll talk about this much later).



df.info()
df.describe()
df.columns


X = df[['', '', column names]]
y = df['Price'] - thing we're trying to predict.

split data into training and test

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0,4, random_state=101)

import linear regression model

lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)
lm.coef_



cdf = pd.DataFram(lm.coef_, X.columns, columns=['Coeff'])


#Predictions

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
- as we are comparing test data to training data, should be closely aligned.

sns.distplot((y_test - predictions))
- should be normally distributed


from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test, predictions)
np.sqrt(metrics.mean_squared_error(y_test, predictions))


-- Interpreting the coefficients:

Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.


# Bias Variance Trade-off


Bias-variance trade-off is the point where we are adding just noise by adding model complexity (flexibility).
Test errors can go up, despite training errors going down.
Model after the bias trade-off begins to overfit.




