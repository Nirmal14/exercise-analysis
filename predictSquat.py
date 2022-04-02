import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import advanced_activations
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np

# def txttoinputData(input_file):
# 	rawStr =''
# 	rawData = []
# 	f = open(input_file, "r")
# 	rawStr = f.read()
# 	rawData= rawStr.split('\n')
# 	data = rawData[slice(1,len(rawData))]
# 	print('----')
# 	print(data[-1])
# 	print('-----')
# 	print(rawData[-1])
# 	dataLabel=[]
# 	for i in range(1,len(rawData)):
# 		x = rawData[i].split(", ")
# 		print(x[-1])
# 		#get labels
# 		dataLabel.append(int(x[-1]))
# 		#get inputs
# 		for j in range(len(x)-1):
# 			x[j] = float(x[j])
# 			data[i-1] = (np.asarray(x[slice(0,-1)]))

# 	data = np.asarray(data)
# #print(len(dataLabel), len(data))
# #print(type(data), type(data[0]))
# # for i in data:
# #   print(i)
# 	return data

def prediction(inputData):
	print("IN prediction")
	data = []
	for i in inputData:
		data.append(np.asarray(i))

	# print("+++++++++++++++++++++\nDATA:")
	# print(data)
	# print("+++++++++++++++++++++\nDATA:")
	# print(type(data))

	data = np.asarray(data)

	# print("+++++++++++++++++++++\nDATA:")
	# print(data)
	# print("+++++++++++++++++++++\nDATA:")
	# print(type(data))

	print("inputs processed")

	output = ''
	output1 = []
	model = load_model('model/squatLinear9790.h5')
	# predictions = model.predict(data, batch_size=10, verbose=0)
	predictions = model.predict_classes(data, batch_size=10, verbose=0)

	for i in range(len(predictions)):
		print()
		right = "Right From"
		ROM = "Bad Form - didn't perform complete range of motion"
		knee = "Bad Form - keep your knees behind your toes at all times"
		back = "Bad Form - keep your back as static as possible and slightly arched"
		rep = "Repitition"
		print("Repitition ",i,": ", end=' ')
		if(predictions[i]==0):
			output = output+','+rep+' '+str(i+1)+": "+right
		elif(predictions[i]==1):
			output = output+','+rep+' '+str(i+1)+": "+ROM
		elif(predictions[i]==2):
			output = output+','+rep+' '+str(i+1)+": "+knee
		elif(predictions[i]==3):
			output = output+','+rep+' '+str(i+1)+": "+back
		elif(predictions[i]==4):
			output = output+','+rep+' '+str(i+1)+": "+knee+" - "+back
	
	return output
	
	