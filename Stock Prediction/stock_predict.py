import numpy as np
import pandas as pd
from dateutil import parser
from pandas_datareader import data
import quandl
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys, time


def getYahoo(symbol, start, end):
	df = data.get_data_yahoo(symbol, start, end)
	df = df.drop(df.columns[[0, 1, 2, 3, 4]], axis=1)
	return df


def getQuandl(symbol, name, start, end):
	df = quandl.get(symbol, trim_start = start, trim_end = end, authtoken="nyZ9we2mw2eRccunvGn5")
	df = df.drop(df.columns[[0, 1, 2, 3, 4]], axis=1)
	return df


def getStocks(date_start, date_end):
	start = parser.parse(date_start)
	end = parser.parse(date_end)

	nasdaq = getYahoo('^IXIC', start, end)
	s_p = getYahoo('^GSPC', start, end)
	djia = getQuandl("YAHOO/INDEX_DJI", 'Djia', start, end)
	hang_seng = getYahoo('^HSI', start, end)
	nikkei = getYahoo('^N225', start, end)
	ftse100 = getYahoo('^FTSE', start, end)
	dax = getYahoo('^GDAXI', start, end)
	asx = getYahoo('^AXJO', start, end)
	gold = getYahoo('GLD', start, end)
	silver = getYahoo('SLV', start, end)
	platinum = getYahoo('PPLT', start, end)
	oil = getYahoo('OIL', start, end)

	return [nasdaq, dax, ftse100, oil]

def getChange(dataset, days):
	for i in range(len(dataset)):
		dataset[i]['%d_pct_change' %i] = dataset[i].pct_change(days)
		dataset[i] = dataset[i].drop(dataset[i].columns[[0]], axis=1)

# need to extrapolate to match NASDAQ
def merge(dataset):

	s1 = dataset[0].join(dataset[1:], how='left')

	s1 = s1.interpolate(method='linear')
	s1 = s1.fillna(s1.mean())
	return s1



def Main():

	date = time.strftime("%Y-%m-%d")
	data_start = '2013-5-5'
	dataset = getStocks('2000-1-26', date)
	getChange(dataset, 1)
	#for i in dataset:
	#	print(i)
	data = merge(dataset)
	y = data[['0_pct_change']].as_matrix()
	y[y >= 0] = 1
	y[y < 0] = 0

	y = y.astype(np.int)
	X = data.ix[:, 1:].as_matrix()
	X[X >= 0] = 1
	X[X < 0] = 0
	X = X.astype(np.int)

	m, n = X.shape

	X_train = X[0:m*.8, :]
	y_train = y[0:m*.8, :]
	
	X_test = X[m*.2:, :]
	y_test = y[m*.2:, :]
	#print(X_test.shape)
	#print(y_test.shape)

	clf = svm.SVC()
	clf.fit(X_train, y_train.ravel())
	predict = clf.predict(X_test)
	#for i in range(len(predict)):
	#	print(predict[i], y_test[i][0])
	print(accuracy_score(y_test, predict))

	#dataset.result[dataset.result >= 0] = 'Up'
	#dataset.result[dataset.result != 'Up'] = 'Down'
	#print(y)
	#print(X.shape)




if __name__ == '__main__':
	Main()
