import csv
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import math
from sklearn import metrics, preprocessing, svm
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

#yolo
#note: all csv's start from most recent to last, except for currencies
indices = []

def parse(filename):
	"""
	parse - parses csv files containing historical prices of stock indices
	filename - the file to parse
	returns:
	array of parsed data
	"""
	# want data at [0] and [4]
	#handle = open(filename)
	data = {}
	with open(filename) as csvfile:
		my_reader = csv.DictReader(csvfile)
		for row in my_reader:
			if filename == 'data/djia.csv':
				date = datetime.datetime.strptime(row['Date'], '%m/%d/%y')
			else:
				date = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
			data[date] = float(row['Close'])
			#print(date.weekday())
	ordered_data = collections.OrderedDict(sorted(data.items()))
	return ordered_data




def evaluate(Features, delta):
	"""
	evaluate - takes in a time t and calculates the difference between two 
	daily prices and sets that new feature
	delta - time to be compared measured in days
	returns:
	new array of data
	"""

	def nextday(listdates, pos):
		"""
		nextday - calculates and determines the next day given a certain
		day. This is to account for weekends & holidays when indices 
		are closed
		index - stock market index name
		day - the given day
		returns - the next day
		"""
		#date = day + datetime.timedelta(days=delta)
		#listdates = list(Features[index].keys())
		return listdates[pos+delta]


	new_features = {}
	for index in Features:
		data = {}
		feature_size = len(Features[index])
		i = 0
		listdates = list(Features[index].keys())
		for day in Features[index]:
			# if on the last day, exit
			
			if i+delta >= feature_size:
				break

			# get the next delta time day
			date = listdates[i+delta]

			# find difference between prices, then normalize it
			data[date] = (Features[index][date] - Features[index][day])/Features[index][day]
			i+=1
		ordered_data = collections.OrderedDict(sorted(data.items()))
		new_features[index] = ordered_data
		#print("created new indices for " + index + "...")
	print('done!')
	return new_features

def normalize(new_features):
	normalized_features = {}
	for index in new_features:
		data = {}
		listvalues = list(new_features[index].values())
		a = np.linalg.norm(listvalues)
		for date in new_features[index]:
			data[date] = new_features[index][date]/a
		ordered_data = collections.OrderedDict(sorted(data.items()))
		normalized_features[index] = ordered_data
		#print("normalized indices for " + index + "...")
	print('done!')
	return normalized_features



def r_squared(correct_stock, predict_stock):
	total_sum = 0
	#date = datetime.datetime(2008, 3, 5)
	index = 0
	for key in correct_stock:
		if key in predict_stock:
			total_sum += ((predict_stock[key] - correct_stock[key])**2)
			index+=1
			#date += datetime.timedelta(days=1)
	#print(total_sum, index)
	total_sum = total_sum/(index)
	return total_sum

def both_equal(new_features, index, date):
	if new_features[index][date] > 0 and new_features['nasdaq'][date] > 0:
		return True
	if new_features[index][date] < 0 and new_features['nasdaq'][date] < 0:
		return True
	return False


def testpls(new_features, index):
	print("Results for " + index + "...")
	# get the training data for DAX
	date = datetime.datetime(2007, 5, 4)
	end_date = datetime.datetime(2011,5,4)
	final_end_date = datetime.datetime(2016, 4,27)
	X = []
	y = []
	X2 = []
	y2 = []
	# training date: 2007 - 2011
	while (True):
		if date == end_date:
			break
		if date in new_features[index] and date in new_features['nasdaq']:
			X.append([new_features[index][date]])
			y.append(1 if new_features['nasdaq'][date] > 0 else -1)
		date += datetime.timedelta(days=1)
	X = np.array(X)
	standard_scaler = StandardScaler()
	X_S = standard_scaler.fit_transform(X)
	X_Scaled = preprocessing.scale(X)
	# testing date: 2011 - 2014
	date = datetime.datetime(2011, 5, 5)
	end_date = datetime.datetime(2014,5,5)
	while (True):
		if date == end_date:
			break
		if date in new_features[index] and date in new_features['nasdaq']:
			X2.append([new_features[index][date]])
			y2.append(1 if new_features['nasdaq'][date] > 0 else -1)
		date += datetime.timedelta(days=1)
	X2 = np.array(X2)
	X_S2 = standard_scaler.transform(X2)
	# fit and predict model
	#model = AdaBoostClassifier()
	#model.fit(X,y)
	model = LinearSVC()
	model.fit(X_S,y)
	expected = y2
	predicted = model.predict(X_S2)
	for i in predicted:
		print(i)
	print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
	#print(model.predict(new_features[index][final_end_date]))
	sys.exit()

def Main():
	folder = 'data'
	indices = ['nasdaq','djia','s&p','oil','asx','dax','gold','platinum',\
	'silver','hang_seng','nikkei','ftse100']
	currency = ['eur','aud','jpy']
	Features = {}
	for i in indices:
		#print("gathering indices for " + i + "...")
		data = parse(folder + '/' + i + '.csv')
		Features[i] = data
	for i in currency:
		#print("gathering indices for " + i + "...")
		data = parse(folder + '/' + i + '.csv')
		Features[i] = data

	print("done!")

	date1 = datetime.datetime(2011, 5, 4)
	new_features = evaluate(Features, 1)
	normalized_features = normalize(new_features)

	# start doin stuff
	date = datetime.datetime(2010, 3, 5)

	testpls(normalized_features, 's&p')
	testpls(normalized_features, 'dax')
	testpls(normalized_features, 'djia')
	testpls(normalized_features, 'hang_seng')
	testpls(normalized_features, 'nikkei')
	testpls(normalized_features, 'jpy')
	testpls(normalized_features, 'gold')
	testpls(normalized_features, 'oil')
	testpls(normalized_features, 'eur')
	testpls(normalized_features, 'ftse100')
	testpls(normalized_features, 'asx')
	testpls(normalized_features, 'aud')


	'''plt.plot(feature1)
	plt.figure()
	plt.plot(feature4)
	plt.figure()
	output_original = np.correlate(feature1, feature1, mode='full')
	output = np.correlate(feature1, feature2, mode='full')
	output2 = np.correlate(feature1, feature3, mode='full')
	output3 = np.correlate(feature1, feature4, mode='full')
	output4 = np.correlate(feature1, feature5, mode='full')
	plt.plot(output_original, 'y')
	plt.plot(output, 'r')
	plt.plot(output2, 'g')
	plt.plot(output3, 'b')
	plt.plot(output4, 'm')
	plt.show()
	'''
	'''
	vector = np.random.normal(0,1,size=1000)
	plt.plot(vector)
	plt.figure()
	vector[::50]+=10
	output = np.correlate(vector,vector,mode='full')
	plt.plot(output)
	plt.show()
	output = np.correlate(feature1,feature2,mode='full')
	print(output)
	plt.plot(output)
	plt.show()'''
if __name__ == '__main__':
    Main()    
