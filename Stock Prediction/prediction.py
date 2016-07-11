import pickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.qda import QDA
import operator
import pandas.io.data
import re
from dateutil import parser
import copy
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

def getStockFromYahoo(symbol, start, end):
	"""
	Gets stock from Yahoo finance
	Computes daily returns based on adj close
	returns pandas dataframe
	"""
	df = pd.io.data.get_data_yahoo(symbol, start, end)
	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + symbol
	df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
	return df

def getStockFromQuandl(symbol, name, start, end):
	"""
	Gets stock from Quandl,
	same as getStock, used for DJIA cuz thats not available on yahoo
	"""

	import quandl
	df = quandl.get(symbol, trim_start = start, trim_end = end, authtoken="nyZ9we2mw2eRccunvGn5")

	df.columns.values[-1] = 'AdjClose'
	df.columns = df.columns + '_' + name
	df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()

	return df

def compileStocks(fout, start_string, end_string):
	"""
	Gets data from yahoo finance & Quandl
	returns list
	"""
	start = parser.parse(start_string)
	end = parser.parse(end_string)
	
	nasdaq = getStockFromYahoo('^IXIC', start, end)
	s_p = getStockFromYahoo('^GSPC', start, end)
	djia = getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start_string, end_string)
	hang_seng = getStockFromYahoo('^HSI', start, end)
	nikkei = getStockFromYahoo('^N225', start, end)
	ftse100 = getStockFromYahoo('^FTSE', start, end)
	dax = getStockFromYahoo('^GDAXI', start, end)
	asx = getStockFromYahoo('^AXJO', start, end)
	gold = getStockFromYahoo('GLD', start, end)
	silver = getStockFromYahoo('SLV', start, end)
	platinum = getStockFromYahoo('PPLT', start, end)
	oil = getStockFromYahoo('OIL', start, end)

	out = pd.io.data.get_data_yahoo(fout, start, end)
	out.columns.values[-1] = 'AdjClose'
	out.columns = out.columns + '_Out'
	out['Return_Out'] = out['AdjClose_Out'].pct_change()
	#return [out, frankfurt]
	return [out, \
			s_p, \
			djia,\
			dax, \
			ftse100, \
			hang_seng, \
			nikkei, \
			asx
			]

def addFeatures(dataframe, adjclose, returns, n):
	"""
	Operates on 2 columns of dataframe
	given return_* computes the return of day i respect to day i-n
	given adjclose_* computes moving average on n days

	"""

	return_n = adjclose[9:] + "Time" + str(n)
	dataframe[return_n] = dataframe[adjclose].pct_change(n)

	roll_n = returns[7:] + "RolMean" + str(n)
	dataframe[roll_n] = pd.rolling_mean(dataframe[returns],n)

def applyReturns(datasets, delta):
	"""
	Applies rolling mean & delayed returns to each dataframe in list
	"""

	for dataset in datasets:
		columns = dataset.columns
		adjclose = columns[-2]
		returns = columns[-1]
		for n in delta:
			addFeatures(dataset, adjclose, returns, n)

	return datasets

def merge(datasets, index, cut):
	"""
	merges datasets in the list
	"""
	#print(datasets)
	# subset gives all other data, excluding the one we're making predictions on
	subset = []
	# get return, time and mean for all datasets excluding first[predictor]
	subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
	#print(subset)
	#outer join, generates union on all columns
	first = subset[0].join(subset[1:], how='outer')

	finance = datasets[0].iloc[:, index:].join(first, how='left')
	# only include values that dates are above the cut
	finance = finance[finance.index > cut]
	return finance


def TimeLag(dataset, lags, delta):
	"""
	apply time lag to return columns selected according to delta.
	Days to lag are contained in the lags list passed as argument

	"""

	dataset.Return_Out = dataset.Return_Out.shift(-1)
	maxLag = max(lags)

	columns = dataset.columns[::(2*max(delta)-1)]
	for column in columns:
		for lag in lags:
			newcolumn = column + str(lag)
			dataset[newcolumn] = dataset[column].shift(lag)

	return dataset.iloc[maxLag:-1,:]

def prepareData(dataset, start_test):
	"""
	generates categorial output column, attach to dataframe
	label the categories & split into train and start_test
	"""

	le = preprocessing.LabelEncoder()
	dataset['result'] = dataset['Return_Out']

	dataset.result[dataset.result >= 0] = 'Up'
	dataset.result[dataset.result != 'Up'] = 'Down'
	dataset.result = le.fit(dataset.result).transform(dataset.result)

	features = dataset.columns[1:-1]

	X = dataset[features]
	y = dataset.result

	X_train = X[X.index < start_test]
	y_train = y[y.index < start_test]

	X_test = X[X.index >= start_test]
	y_test = y[y.index >= start_test]

	return X_train, y_train, X_test, y_test

def performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel):
	"""
	performs classification on daily returns using several algorithms (method).
	method --> string algorithm
	parameters --> list of parameters passed to the classifier (if any)
	fout --> string with name of stock to be predicted
	savemodel --> boolean. If TRUE saves the model to pickle file
	"""
	
	if method == 'RF':
		return performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
		
	elif method == 'KNN':
		return performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
	
	elif method == 'SVM':
		return performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)
	
	elif method == 'ADA':
		return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel)

def performSVMClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
	"""
	SVM binary classification
	"""
	clf = QDA()
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)
	return accuracy

def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
	"""
	Ada boosting binary classification
	"""
	clf = AdaBoostClassifier()
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)

	return accuracy

def performRFClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
	"""
	Random Forest Binary Classification
	"""
	clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	
	return accuracy

def performKNNClass(X_train, y_train, X_test, y_test, parameters, fout, savemodel):
	"""
	KNN binary Classification
	"""
	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)
	
	return accuracy

def FeatureSelection(maxdeltas, maxlags, fout, cut, start_test, path_datasets, savemodel, method, folds, parameters):
	"""
	Performs Feature selection for a specific algorithm
	"""
	x,y = [], []
	for maxlag in range(3, maxlags + 2):
		lags = range(2, maxlag) 
		print('')
		print('=============================================================')
		print('Maximum time lag applied', max(lags))
		print('')
		for maxdelta in range(3, maxdeltas + 2):
			datasets = copy.deepcopy(path_datasets)
			#datasets = loadDatasets(path_datasets, fout)
			delta = range(2, maxdelta)
			print('Delta days accounted: ', max(delta))
			datasets = applyReturns(datasets, delta)
			finance = merge(datasets, 6, cut)
			print('Size of data frame: ', finance.shape)
			finance = finance.interpolate(method='linear')
			finance = finance.fillna(finance.mean())
			finance = TimeLag(finance, lags, delta)
			print('Size of data frame after feature creation: ', finance.shape)
			X_train, y_train, X_test, y_test  = prepareData(finance, start_test)
			accuracy = performClassification(X_train, y_train, X_test, y_test, method, parameters, fout, savemodel)
			#accuracy = performCV(X_train, y_train, folds, method, parameters, fout, savemodel)
			print("Mean Accuracy for (delta = ", max(delta), "): ", accuracy)
			print('')
			x.append(max(delta))
			y.append(accuracy)
	plt.plot(x,y, label=method)
	plt.xlabel('delta time difference')
	plt.ylabel('Accuracy (%)')




def Main():
	fout = '^IXIC'
	test_data_start = '2013-5-5'
	dataset = compileStocks(fout, '2000-5-5', '2014-5-5')
	delta = 5
	FeatureSelection(delta,2,fout, '2006-1-19', test_data_start, dataset, False, 'RF', 10, [0,1])
	#FeatureSelection(delta,2,fout, '2006-1-19', test_data_start, dataset, False, 'ADA', 10, [0,1])
	#FeatureSelection(delta,2,fout, '2006-1-19', test_data_start, dataset, False, 'KNN', 10, [0,1])
	#FeatureSelection(delta,2,fout, '2006-1-19', test_data_start, dataset, False, 'SVM', 10, [0,1])
	plt.legend()
	plt.show()
if __name__ == '__main__':
	Main()