import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from predictSquat import prediction
# from predictSquat import txttoinputData
from tornado.options import define, options
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import statistics 
from scipy import signal as ss
from scipy.signal import find_peaks, peak_prominences, peak_widths


def bestApproximate(x, y, points): 
	n = len(x)
	sum_x = 0
	sum_y = 0
	sum_xy = 0
	sum_x2 = 0
	maxi = y[0]
	mini = y[-1]
	flag=0

	if(y[0]<y[-1]):
		maxi = y[-1]
		mini = y[0]

	for i in range (n): 
		sum_x += x[i] 
		sum_y += y[i] 
		sum_xy += x[i] * y[i] 
		sum_x2 += pow(x[i], 2) 

	m = (float)((n * sum_xy - sum_x * sum_y) 
			/ (n * sum_x2 - pow(sum_x, 2))); 
			
	c = (float)(sum_y - m * sum_x) / n;
	
	Y = np.asarray([])
	X = np.asarray([])
	tempX = []
	index = []
	for i in range(1, len(x)-1):
		temp = m*x[i] + c
		if  mini < temp < maxi:
			tempX.append(x[i])
			index.append(i)
			flag=flag+1

	# put start values
	X=np.append(X,x[0])
	Y=np.append(Y,y[0])

	if (flag>=2):
		# addvalues between start and tempX[0]
		X = np.append(X, x[slice(0, index[0])])
		Y = np.append(Y, y[slice(0, index[0])])
	else:
		tempX=x
	
	z = (abs(tempX[0]-tempX[-1]))/points
	temp=0
	j=tempX[0]+z
	while j<tempX[-1]:
		temp = m*j + c
		X=np.append(X,j)
		Y=np.append(Y,temp)
		j = j+z
	
	if(flag>=2):
		# add values between tempX[-1] and end
		X = np.append(X, x[slice(index[-1]+1, len(x))])
		Y = np.append(Y, y[slice(index[-1]+1, len(y))])

	#put end values
	X=np.append(X,x[-1])
	Y=np.append(Y,y[-1])


	#remove extra data points starting from centre
	remove = len(X)-points

	for i in range(remove):
		k = int((len(X)/2) + (((-1)**i)*1)) 
		X = np.delete(X, k, 0)
		Y = np.delete(Y, k, 0)
	
	return([X, Y])

class MainHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		self.set_header("Access-Control-Allow-Origin", "*")
		self.set_header("Access-Control-Allow-Headers", "x-requested-with")
		self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

	def post(self):
		kneeval = self.get_argument('kneeval',True)
		ankleVal1 = self.get_argument('ankleVal1',True)
		ankleVal2 = self.get_argument('ankleVal2',True)
		list_1 = []
		list_2 = []
		list_3 = []
		# print(kneeval[0],kneeval[1])
		# print(kneeval[-2])
		# print(kneeval[-1])
		for i in kneeval[slice(1,-1)].split(','):
			list_1.append(float(i))
		
		for i in ankleVal1[slice(1,-1)].split(','):
			list_2.append(float(i))
		
		for i in ankleVal2[slice(1,-1)].split(','):
			list_3.append(float(i))
		ks = 5
		y1=ss.medfilt(list_1, kernel_size=ks)
		y2=ss.medfilt(list_2, kernel_size=ks)
		y3=ss.medfilt(list_3, kernel_size=ks)

		y1n = np.array([])
		y1n=ss.medfilt(y1, kernel_size=ks)
		y2n=ss.medfilt(y2, kernel_size=ks)
		y3n=ss.medfilt(y3, kernel_size=ks)

		x1n=[]
		for n in range(0,len(y1n)):
			x1n.append(n)

		print("Type of x1 and y1n: ",type(x1n), type(y1n))

		y1Max = find_peaks(y1n)
		# print("---------------------------------")
		# print("Type of y1Max: ",type(y1Max))
		# print("Type of y1Max[0]: ",type(y1Max[0]))
		# print("y1Max: ",y1Max)

		tempMax = peak_prominences(y1n, y1Max[0])
		# print("---------------------------------")
		# print("Type TempMax: ", type(tempMax))
		# print("Temp Max:\n",tempMax)
		yMax=[]
		tempMaxNew=[]
		tempMaxP=np.array([])
		tempMaxL=np.array([])
		tempMaxR=np.array([])
		# yMax.append(0)
		n=0
		while n<len(tempMax[0]):
			if tempMax[0][n] > 0.4:	
				yMax.append(y1Max[0][n])
				tempMaxP=np.append(tempMaxP,tempMax[0][n])
				tempMaxL=np.append(tempMaxL,tempMax[1][n])
				tempMaxR=np.append(tempMaxR,tempMax[2][n])
			n=n+1
		# append the last peak value also to yMax
		# yMax.append(y1Max[0][len(y1Max[0])-1])
		tempMaxNew.append(tempMaxP)
		tempMaxNew.append(tempMaxL.astype(int))
		tempMaxNew.append(tempMaxR.astype(int))
		tempMaxNew = tuple(tempMaxNew)
		# print("---------------------------------")
		# print("yMax: ",yMax)
		# print("type: ",type(yMax))
		# print("---------------------------------")
		# print("tempMaxNew: \n", tempMaxNew)
		# print("type: ", type(tempMaxNew))
		peak=np.asarray(yMax)
		prominenceMax=tempMaxP

		combineMax=[]
		combineMax.append(tempMaxL.astype(int))
		combineMax.append(tempMaxR.astype(int))
		combineMax=np.unique(combineMax)
		# print("---------------------------------")
		# print("combineMax: ",combineMax)

		#calculating inverse
		y1nIn = np.arccos(y1n)
		y2nIn = np.arccos(y2n)
		y3nIn = np.arccos(y3n)	

		#find troughs from find_peaks
		y1Min = find_peaks(y1nIn)
		# print("y1Min: ",y1Min[0])

		tempMin = peak_prominences(y1nIn, y1Min[0])
		# print("Temp Min:\n",tempMin)

		# trough=np.asarray(y1Min[0])
		# prominenceMin=np.asarray(tempMin[0])

		# yMin has only relavent peaks. ie peaks with prominance > 0.5
		yMin=[]
		tempMinNew=[]
		tempMinP=np.array([])
		tempMinL=np.array([])
		tempMinR=np.array([])
		# yMin.append(0)
		n=0
		while n<len(tempMin[0]):
			if tempMin[0][n] > 0.4:
				yMin.append(y1Min[0][n])
				tempMinP=np.append(tempMinP, tempMin[0][n])
				tempMinL=np.append(tempMinL, tempMin[1][n])
				tempMinR=np.append(tempMinR, tempMin[2][n])
			n=n+1
		# append the last peak value also to y1Max
		# yMin.append(y1Min[0][len(y1Min[0])-1])
		# print("yMin: ",yMin)
		# print("type: ",type(yMin))

		trough=np.asarray(yMin)
		prominenceMin=tempMinP

		combineMin=[]
		combineMin.append(tempMinL.astype(int))
		combineMin.append(tempMinR.astype(int))
		combineMin=np.unique(combineMin)
		# print("combineMin: ",combineMin)

		# #cal peak bases
		# peakBase=[]
		# for n in combineMax:
		# 	peakBase.append(y1n[n])

		# #cal trough base
		# troughBase=[]
		# for n in combineMin:
		# 	troughBase.append(y1n[n])

		# cal one cycle
		combine = np.union1d(combineMin, combineMax)
		# print("combine: \n", combine)
		# print("type: ",type(combine))
		cycle=[]
		cycleX=[]

		for n in range(len(combine)):
			if(len(cycle)>0):
				if( abs( (y1n[combine[n]]) - cycle[len(cycle)-1] ) < 0.1 ):
					continue
				# if ( (combine[n]-cycleX[len(cycleX)-1]) < 2):
				# 	continue

			cycle.append(y1n[combine[n]])
			cycleX.append(combine[n])


		y1n = np.asarray(y1n)
		combine = np.asarray(combine)
		# print(type(y1n), type(combine))
		# print("combine: ",combine)
		# print("impPoints: ",y1n[combine])

		# trimming the data futher in the begining and the end
		if(cycle[0] > cycle[1]):
			# print("++++++++")
			# print('1')
			del cycle[0]
			del cycleX[0]
			del yMax[0]

		temp = len(cycle)
		if(cycle[temp-1] > cycle[temp-2]):
			# print("++++++++")
			# print('1')
			del cycle[temp-1]
			del cycleX[temp-1]
			del yMax[-1]

		temp = len(cycleX)
		n=0
		while n<temp-1:
			if(cycle[n]<cycle[n+1] and cycle[n+2]<cycle[n+1]):
				n=n+2
				continue
			maxi = n
			indexMaxi = cycleX[n]
			if(cycle[n+1] > cycle[maxi]):
				maxi = n+1
				indexMaxi = cycleX[n+1]
			del(cycle[maxi])
			del(cycleX[maxi])
			n = n+2
			temp = len(cycleX)

		# best line fiting part
		points = 200
		newCycleY=[]
		newCycleX=[]
		newCycleY1=[]
		newCycleX1=[]
		newCycleY2=[]
		newCycleX2=[]
		newCycleX3=[]
		newCycleY3=[]
		n=0
		while n < len(cycleX)-2:
			x = np.asarray([])
			y1_fit = np.asarray([])

			x1 = x1n[slice(cycleX[n], cycleX[n+1]+1)]
			x2 = x1n[slice(cycleX[n+1], cycleX[n+2]+1)]
			# y1n - leg
			y1fit1 = y1n[slice(cycleX[n], cycleX[n+1]+1)]
			y1fit2 = y1n[slice(cycleX[n+1], cycleX[n+2]+1)]
			# y1n - knee
			y2fit1 = y2n[slice(cycleX[n], cycleX[n+1]+1)]
			y2fit2 = y2n[slice(cycleX[n+1], cycleX[n+2]+1)]
			# y1n - back
			y3fit1 = y3n[slice(cycleX[n], cycleX[n+1]+1)]
			y3fit2 = y3n[slice(cycleX[n+1], cycleX[n+2]+1)]

			# y1n - leg
			#for first half
			temp = bestApproximate(x1, y1fit1, points)
			x = np.append(x, temp[0])
			y1_fit = np.append(y1_fit, temp[1])
			#for seconf half 
			temp = bestApproximate(x2, y1fit2, points)
			x = np.append(x, temp[0])
			y1_fit = np.append(y1_fit, temp[1])
			#store
			newCycleX1.append(x)
			newCycleY1.append(y1_fit)
			
			# y2n - knee
			#for first half
			temp = bestApproximate(x1, y2fit1, points)
			x = np.append(x, temp[0])
			y1_fit = np.append(y1_fit, temp[1])
			#for seconf half 
			temp = bestApproximate(x2, y2fit2, points)
			x = np.append(x, temp[0])
			y1_fit = np.append(y1_fit, temp[1])
			#store
			newCycleX2.append(x[slice(points*2, points*4)])
			newCycleY2.append(y1_fit[slice(points*2, points*4)])
			
			# y3n - back
			#for first half
			temp = bestApproximate(x1, y3fit1, points)
			x = np.append(x, temp[0])
			y1_fit = np.append(y1_fit, temp[1])
			#for seconf half 
			temp = bestApproximate(x2, y3fit2, points)
			x = np.append(x, temp[0])
			y1_fit = np.append(y1_fit, temp[1])
			#store
			newCycleX3.append(x[slice(points*4, points*6)])
			newCycleY3.append(y1_fit[slice(points*4, points*6)])
			
			#store TOGETHER
			newCycleX.append(x)
			newCycleY.append(y1_fit)

			n = n+2

		#printing newCycle 
		print("+++++++++++++++++++++++++++++")
		# print(newCycleX1[0])
		print(len(newCycleX))
		print("+++++++++++++++++++++++++++++")
		# print(newCycleY1[0])
		print(len(newCycleY[0]), len(newCycleY1[0]), len(newCycleY2[0]), len(newCycleY3[0]))

		print("check: ", len(newCycleY))
		print("+++++++++++++++++")
		print(newCycleY)
		output=''
		if(len(newCycleY[0]) != points*6):
			output="die"
		else:
			output=prediction(newCycleY)

		
		# formatOutput = ''
		# for i in output.split(','):
		# 	formatOutput = formatOutput+"\n"+i
		#print(type(data))
		#print(self.get_argument('ankleVal1',True))
		#print(self.get_argument('ankleVal2',True))
		
		self.write(output)


def main():
	tornado.options.parse_command_line()
	application = tornado.web.Application([(r"/", MainHandler)])
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(8888)
	tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
	main()


