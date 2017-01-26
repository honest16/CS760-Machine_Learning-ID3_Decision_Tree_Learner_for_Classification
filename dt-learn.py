# Importing standard libraries
import sys
import math
import random

# Node Class
class Node(object):
	def __init__(self):
		self.parentSplitFeat = None
		self.parentSplitVal = None
		self.parentSplitOp = None
		self.parentLabel = None
		self.splitFeature = None
		self.splitFeaType = None
		self.posNegVec = None
		self.branchVals =None
		self.children = None
		self.label = None
		self.posNegHere = None

# Class to print test results		
class PrintTestResults(object):
	def __init__(self,actualPred,numCorr,numTest):
		print '<Predictions for the Test Set Instances>'
		for i in range(numTest):
			print str(i+1)+': '+ 'Actual: ' + actualPred[0][i] + ' Predicted: '+ actualPred[1][i]
		print 'Number of correctly classified: '+str(numCorr) + ' Total number of test instances: '+str(numTest)
		
# PrintTree Class
class PrintTree(object):
	def __init__(self,node,printprefix):
		# If leaf node 
		if node.children == None:
			if node.parentSplitOp == '=':
				print printprefix + node.parentSplitFeat + ' '+ node.parentSplitOp + ' '+ str(node.parentSplitVal) + ' ' + '['+str(node.posNegHere[0])+ ' ' + str(node.posNegHere[1])+']: ' + node.label
			else:
				a = float(node.parentSplitVal)
				print printprefix + node.parentSplitFeat + ' '+ node.parentSplitOp + ' '+ '%.6f'%a + ' ' + '['+str(node.posNegHere[0])+ ' ' + str(node.posNegHere[1])+']: ' + node.label
			return
		# If non leaf node but not root
		elif node.parentSplitFeat !=None:
			if  node.parentSplitOp == '=':
				
				print printprefix + node.parentSplitFeat + ' '+ node.parentSplitOp + ' '+ str(node.parentSplitVal) + ' ' + '['+str(node.posNegHere[0])+ ' ' + str(node.posNegHere[1])+']' 
			else:
				a = float(node.parentSplitVal)
				print printprefix + node.parentSplitFeat + ' '+ node.parentSplitOp + ' '+ '%.6f'%a + ' ' + '['+str(node.posNegHere[0])+ ' ' + str(node.posNegHere[1])+']' 
			printprefix = '|	'+ printprefix
			for child in node.children:
				PrintTree(child,printprefix)
		# If root
		elif node.parentSplitFeat == None:
			for child in node.children:
				PrintTree(child,printprefix)

				
# To predict labels for test set instances
class Classify(object):
	def __init__(self,root,test_set):
		self.actualPred =[]
		self.child = None
		actual = []
		pred = []
		featIndMap = {}
		featTypeMap = {}
		for k,v in test_set.attributes.iteritems():
			attrIndex = k
			for k1,v1 in test_set.attributes[k].iteritems():
				feature = k1
				if v1 == 'numeric':
					type = 'numeric'
				elif v1 == 'real':
					type = 'real'
				else:
					type = 'nomimal'
			featIndMap[feature] = attrIndex
			featTypeMap[feature] =  type
			
		for item in test_set.instances:
			actual.append(item[-1])
			mp = makePrediction(root, featIndMap,featTypeMap, item)		
			pred.append(mp.currPred)
		
			
		self.actualPred = [actual,pred]
		
		
		
class makePrediction(object):
	def __init__(self,node,featIndMap,featTypeMap,item):
		
		if node.children == None:
			self.currPred = node.label
		
		else:
			
			attrIndex = featIndMap[node.splitFeature]  
		
			if featTypeMap[node.splitFeature] == 'numeric' or featTypeMap[node.splitFeature] == 'real':
				insFeatVal = float(item[attrIndex])
		
				if insFeatVal <= float(node.branchVals):
					child = node.children[0]
					mp = makePrediction(child,featIndMap,featTypeMap,item)
					self.currPred = mp.currPred
				else:
					child = node.children[1]
					mp = makePrediction(child,featIndMap,featTypeMap,item)
					self.currPred = mp.currPred
					
			elif featTypeMap[node.splitFeature] == 'nomimal':
				insFeatVal = item[attrIndex]
				for i in range(0,len(node.branchVals)):
					if insFeatVal == node.branchVals[i]:
						child = node.children[i]
						mp = makePrediction(child,featIndMap,featTypeMap,item)
						self.currPred = mp.currPred
						break
			
			
# Class for data contents
class dataContents(object):
	# Defining instance members attributes and instances
	def __init__(self,name):
		self.attributes = None
		self.getAttributes(name)
		self.instances = None
		self.getInstances(name)
		
		
	# Method to get attributes from file 
	def getAttributes(self,name):
		attrDict = {}
		numAttr = 0
		f = open(name)
		for line in f:
			if line[0] != '@':
				break
			elif '@relation' in line or '@data' in line:
				continue
			else:
				newline = line[len('@attribute'):]
				lineList = newline.split("'")
				lineList[2] =  lineList[2][0:-1]
				if '{' in lineList[2] or '}' in lineList[2]:
					for i in range(len(lineList[2])):
						if lineList[2][i] == '{':
							startInd = i
						if lineList[2][i] == '}':
							endInd = i
					attrStr = lineList[2][0:startInd] + lineList[2][startInd+1: endInd] + lineList[2][endInd+1:]
					attributeVals = [elem.strip() for elem in attrStr.split(",")]
				else:
					attributeVals = lineList[2].strip()
					
				attrDict[numAttr] = {lineList[1]:attributeVals}
				numAttr += 1

		self.attributes = attrDict
		f.close()
		#print self.attributes

		
	# Method to get instances from file
	def getInstances(self,name):
		f  = open(name)
		num = 0
		instanceList = []
		for line in f:
			if line[0] == '@':
				continue
			elif line == None:
				break
			else:
				num += 1
				instanceList.append((line.strip().split(',')))
		self.instances = instanceList
		f.close()

		
# Method to get contents from file name
def getContents(name):
	contents = dataContents(name)
	return contents

# Class Decision Tree
class decisionTree(object):

	# Defining class variables
	def __init__(self):
		self.train_set = None
		self.test_set = None
		self.m = None
		# Obtaining command line arguments
		self.obtainCLIArgs()
		D = [i for i in range(0,len(self.train_set.instances))]
		numAttrs = len(self.train_set.attributes)
		labelDict =  self.train_set.attributes[numAttrs-1]
		labelVals =[]
		for k, v in labelDict.iteritems():
			labelVals = v
		
		
		instInd = 0
		numPosD = 0
		numNegD = 0
		for item in self.train_set.instances:
			if instInd in D:
				if item[-1] == labelVals[0]:
					numPosD += 1
				if item[-1] == labelVals[1]:
					numNegD += 1
				instInd += 1
			else:
				instInd += 1 
				continue
		

		root = Node()
		root.parentSplitFeat = None
		root.parentSplitVal = None
		root.parentSplitOp = None
		root.parentLabel = None
		root.posNegHere = [numPosD,numNegD]
		rootWithTree = self.makeSubTree(D,root)
		PrintTree(rootWithTree, '')
		cls = Classify(rootWithTree,self.test_set)
		numCorr = 0
		numTest = len(cls.actualPred[0])
		for i in range(numTest):
			if cls.actualPred[0][i] == cls.actualPred[1][i]:
				numCorr += 1

		PrintTestResults(cls.actualPred,numCorr,numTest)

		
	# Method to obtain data from command line arguments
	def obtainCLIArgs(self):
		
		trainFileName =  sys.argv[1]
		
		testFileName = sys.argv[2]
		
		self.m = int(sys.argv[3])
		
		self.train_set = getContents(trainFileName)  
		self.test_set = getContents(testFileName)

		
	# Method to find best split
	def findBestSplit(self, C, D):
	
		S = []
		if len(D) == 0:
			return S
		numPosD = 0
		numNegD = 0
		numAttrs = len(self.train_set.attributes)
		labelDict =  self.train_set.attributes[numAttrs-1]
	
		labelVals =[]
		for k, v in labelDict.iteritems():
			labelVals = v
		
		instInd = 0
		for item in self.train_set.instances:
			if instInd in D:
				if item[-1] == labelVals[0]:
					numPosD += 1
				if item[-1] == labelVals[1]:
					numNegD += 1
				instInd += 1
			else:
				instInd += 1 
				continue
				
				
	
		numPosD = float(numPosD)
		numNegD = float(numNegD)
		numTotD = numPosD + numNegD
			
		posFracD = numPosD/numTotD
		negFracD = numNegD/numTotD
		
		if posFracD > 0 and negFracD > 0:
			HDY = -(posFracD)*(math.log(posFracD,2)) - (negFracD)*(math.log(negFracD,2))
		elif posFracD > 0 and negFracD == 0:
			HDY = -(posFracD)*(math.log(posFracD,2)) 
		elif posFracD == 0 and negFracD > 0:	
			HDY = - (negFracD)*(math.log(negFracD,2)) 
			
		#best nomimal feature to split on
		bestNomAttr = None
		bestNomInfGain = -100
		bestNomPosNegVec = None
		bestNomAttrInd = None
		#best numeric feature to split on and the corresponding threshold
		bestNumAttr = None
		bestNumInfGain = -100 
		bestNumPosNegVec = None
		bestNumThresh = None
		bestNumAttrInd = None
		
		for k,v in C.iteritems():
			
			attrName= k
			for key,val in self.train_set.attributes.iteritems():
				for key1, val1 in val.iteritems():
					if key1 == attrName:
						attrIndex = key
						
			for kl2, vl2 in C[k].iteritems():
				attrNumChild =  kl2
				attrVals = vl2
				
				if not attrVals:
					continue
				if type(attrVals[0]) is str:

					posNegVec = []
					for i in range(0,attrNumChild):
						valList =[0,0]
						posNegVec.append(valList)
				
					instInd = 0			
					for item in self.train_set.instances:
						if instInd in D:
							if item[-1] == labelVals[0]:
								# What is the nominal value that the attrIndex is taking
								nomVal = item[attrIndex]
								# What list would that correspond to?
								for i in range(0,attrNumChild):
									if nomVal ==attrVals[i]:
										corri = i
								# Increment the first element
								posNegVec[corri][0] += 1

							 
							if item[-1] == labelVals[1]:
								# What is the nominal value that the attrIndex is taking
								nomVal = item[attrIndex]
								# What list would that correspond to?
								for i in range(0,attrNumChild):
									if nomVal ==attrVals[i]:
										corri = i
								# Increment the sec element
								posNegVec[corri][1] += 1

							instInd += 1
						else:
							instInd += 1
							continue
						

					HDYS = 0
					try:
						for i in range(len(posNegVec)):
							numPosBr = float(posNegVec[i][0])
							numNegBr = float(posNegVec[i][1])
							numInBr = numPosBr + numNegBr
							if numPosBr == 0 and numNegBr == 0:
								continue
							else:
								posFracBr = numPosBr/numInBr
								negFracBr = numNegBr/numInBr

							if posFracBr > 0 and negFracBr > 0:
								HDYgB = -(posFracBr)*(math.log(posFracBr,2))-(negFracBr)*(math.log(negFracBr,2))
							if posFracBr >0  and negFracBr ==0:
								HDYgB = -(posFracBr)*(math.log(posFracBr,2))
							if posFracBr == 0 and negFracBr > 0:
								HDYgB = -(negFracBr)*(math.log(negFracBr,2))
			
							HDYS += (numInBr/numTotD)*HDYgB
					except ZeroDivisionError: 
						#print 'Problem computing HDYgB'
						#print attrName, attrIndex, posNegVec
						continue
					except ValueError:
						#print 'Problem computing HDYgB'
						#print attrName, attrIndex, posNegVec
						continue
					

					InfGain = HDY - HDYS
					
					if InfGain > bestNomInfGain:
						bestNomInfGain = InfGain
						bestNomAttr = attrName
						bestNomPosNegVec = posNegVec
						bestNomAttrInd = attrIndex
					
					if InfGain == bestNomInfGain and attrIndex < bestNomAttrInd:
						bestNomInfGain = InfGain
						bestNomAttr = attrName
						bestNomPosNegVec = posNegVec
						bestNomAttrInd = attrIndex		

				
		# Compute Info Gain on Nominal Features
				if type(attrVals[0]) is float:

					posNegVecBest = [[0,0],[0,0]]
					thresholdBest = 100000 
					InfGainBest = -100 
					
					for thInd in range(0,len(attrVals)):
						threshold = attrVals[thInd]
						posNegVec = [[0,0],[0,0]]				
						insInd = 0
						
						for item in self.train_set.instances:
							if insInd in D:
								if item[-1] == labelVals[0]:
									# What is the nominal value that the attrIndex is taking
									featVal = float(item[attrIndex])
								
									# What list would that correspond to?
									if featVal <= threshold:
										# Left branch and label = positive
										posNegVec[0][0] += 1
									else:
										# Right branch and label = positive
										posNegVec[1][0] += 1
									
								# Label = Negative		
								if item[-1] == labelVals[1]:
									featVal = float(item[attrIndex])
									if featVal <= threshold:
										# Left branch and label = negative
										posNegVec[0][1] += 1
									else:
										# Right branch and label = negative
										posNegVec[1][1] += 1
										
								insInd += 1
							else:
								insInd += 1
								continue
									
						
						HDYSt = 0
						
						try:
							for i in range(len(posNegVec)):
								numPosBr = float(posNegVec[i][0])
								numNegBr = float(posNegVec[i][1])
								numInBr = numPosBr + numNegBr
								posFracBr = numPosBr/numInBr
								negFracBr = numNegBr/numInBr
								if posFracBr > 0 and negFracBr > 0:
									HDYgB = -(posFracBr)*(math.log(posFracBr,2))-(negFracBr)*(math.log(negFracBr,2))
								if posFracBr > 0 and negFracBr == 0:
									HDYgB = -(posFracBr)*(math.log(posFracBr,2))
								if posFracBr == 0 and negFracBr > 0:
									HDYgB = -(negFracBr)*(math.log(negFracBr,2))
								HDYSt += (numInBr/numTotD)*HDYgB
						except ValueError:
							continue
						except ZeroDivisionError: 
							continue
						
						InfGain = HDY - HDYSt
						if InfGain > InfGainBest:
							InfGainBest = InfGain
							posNegVecBest = posNegVec
							thresholdBest = threshold
						if InfGain == InfGainBest:
							if threshold >= thresholdBest: 
								pass
							else:
								thresholdBest = threshold
								InfGainBest = InfGain
								posNegVecBest = posNegVec
								
							
					if InfGainBest > bestNumInfGain:
						bestNumInfGain = InfGainBest
						bestNumAttr = attrName
						bestNumPosNegVec = posNegVecBest
						bestNumThresh = thresholdBest
						bestNumAttrInd = attrIndex
					if InfGainBest == bestNumInfGain and attrIndex < bestNumAttrInd:
						bestNumInfGain = InfGainBest
						bestNumAttr = attrName
						bestNumPosNegVec = posNegVecBest
						bestNumThresh = thresholdBest
						bestNumAttrInd = attrIndex		
		
		S = []
		if bestNomInfGain > bestNumInfGain:
			S = [bestNomInfGain, bestNomAttr, bestNomAttrInd, bestNomPosNegVec]
		elif bestNomInfGain < bestNumInfGain:
			S = [bestNumInfGain, bestNumAttr, bestNumAttrInd, bestNumPosNegVec, bestNumThresh]
		elif bestNomInfGain == bestNumInfGain:
			if bestNomAttrInd <= bestNumAttrInd:
				S = [bestNomInfGain, bestNomAttr, bestNomAttrInd, bestNomPosNegVec]
			if bestNomAttrInd > bestNumAttrInd:
				S = [bestNumInfGain, bestNumAttr, bestNumAttrInd, bestNumPosNegVec, bestNumThresh]
				
		
		return S

	
	def makeSubTree(self,D,node):
		C = self.determineCandidateSplits(D)
		
		
		S = self.findBestSplit(C,D)
		
				
		
		crit1 = False
		crit2 = False
		crit3 = False
		crit4 = False
		
		# If either all instances are + or all are -
		if node.posNegHere[1] == 0 or node.posNegHere[0] == 0:
			crit1 = True
		
		# Number of instances fewer than m
		if len(D) < self.m:
			crit2 = True
		
		# If no feature has positive information gain/ no best split	
		if len(S) == 0 or S[0] <0: 	
			crit3 = True
			
		# No remaining candidate splits
		if not C: 
			crit4 = True
		
		# TODO: See if this helps
		stopping_criteria = False
		if crit1 or crit2 or crit3 or crit4:
			stopping_criteria = True
		
	
		if stopping_criteria:
	
			node.splitFeature = None
			node.splitFeaType = None
			node.posNegVec = None
			node.branchVals =None
			node.children = None
	
			numAttrs = len(self.train_set.attributes)
			labelDict =  self.train_set.attributes[numAttrs-1]
	
			labelVals =[]
			for k, v in labelDict.iteritems():
				labelVals = v
		
	
			if node.posNegHere[0] > node.posNegHere[1] and len(D)>0:
				node.label = labelVals[0]
			elif node.posNegHere[0] < node.posNegHere[1] and len(D)>0:
				node.label = labelVals[1]
			elif node.posNegHere[0] == node.posNegHere[1] and len(D)>0:
				node.label = node.parentLabel
			elif len(D)==0:
				node.label = node.parentLabel
		
			return node
							
			
		else:
		
			node.splitFeature = S[1]
			if len(S) == 4:
				node.splitFeaType = 'nominal'
				node.branchVals = self.train_set.attributes[S[2]][S[1]]
			elif len(S) == 5:
				node.splitFeaType = 'numeric' or node.splitFeaType =='real'
				node.branchVals =float(S[-1])
			node.posNegVec = S[3]
			
			
			if node.splitFeaType == 'numeric' or node.splitFeaType =='real':
				numCh = 2
				
				numAttrs = len(self.train_set.attributes)
				labelDict =  self.train_set.attributes[numAttrs-1]
		
				labelVals =[]
				for k, v in labelDict.iteritems():
					labelVals = v
				
				
				leftCh = Node()
				leftCh.parentSplitFeat = node.splitFeature
				leftCh.parentSplitVal = float(node.branchVals)
				leftCh.parentSplitOp = '<='
				
				if node.posNegHere[0] >= node.posNegHere[1]: 
		
					leftCh.parentLabel = labelVals[0] 
				else:
					leftCh.parentLabel = labelVals[1]
				
				
				leftCh.posNegVec = None 
				leftCh.branchVals =None
				leftCh.children = None
	
				leftCh.posNegHere = node.posNegVec[0]
				
				rightCh = Node()
				rightCh.parentSplitFeat = node.splitFeature
				rightCh.parentSplitVal = float(node.branchVals)
				rightCh.parentSplitOp = '>'
				
					
				if node.posNegHere[0] >= node.posNegHere[1]: 
					rightCh.parentLabel = labelVals[0] 
				else:
					rightCh.parentLabel = labelVals[1]
					
				
				rightCh.posNegVec = None 
				rightCh.branchVals =None
				rightCh.children = None
				rightCh.posNegHere = node.posNegVec[1]
				
				
				# Determing Dleft and Dright
				# This is a numeric feature
				# instances with numeric feature value < = node.branchVals will go to Dleft
				instInd = 0
				Dleft = []
				Dright = []
				for item in self.train_set.instances:
					if instInd in D:
						if float(item[S[2]]) <= node.branchVals:
							Dleft.append(instInd)
						else:
							Dright.append(instInd)
						instInd += 1
					else:
						instInd += 1 
						continue
		
				
				
				# rootWithTree = self.makeSubTree(D,root)
				leftChNew = self.makeSubTree(Dleft, leftCh)
				leftChNew.parentSplitVal = float(leftChNew.parentSplitVal)
				rightChNew = self.makeSubTree(Dright, rightCh)
				rightChNew.parentSplitVal = float(rightChNew.parentSplitVal)
				node.children = [leftChNew, rightChNew]
				
			elif node.splitFeaType == 'nominal':
				node.children = []
				numCh = len(node.branchVals)
				Dks = []
				for i in range(0,numCh):
					Di = []
					Dks.append(Di)

				instInd = 0
				for item in self.train_set.instances:
					if instInd in D:
						
						for i in range(0,numCh):
							if item[S[2]] == node.branchVals[i]:
								Dks[i].append(instInd)
								break
						instInd += 1
					else:
						instInd += 1 
						continue
		
				
				
				
				
				for i in range(0,numCh):
					ci = Node()
					ci.parentSplitFeat = node.splitFeature
					ci.parentSplitVal = node.branchVals[i]
					ci.parentSplitOp = '='
					
					
					numAttrs = len(self.train_set.attributes)
					labelDict =  self.train_set.attributes[numAttrs-1]
					
					labelVals =[]
					for k, v in labelDict.iteritems():
						labelVals = v
					
					if node.posNegHere[0] >= node.posNegHere[1]:  
					
						ci.parentLabel = labelVals[0] 
					else:
						ci.parentLabel = labelVals[1]
					
					
	
					ci.posNegHere = node.posNegVec[i]
					ciNew = self.makeSubTree(Dks[i], ci)
										
					node.children.append(ciNew)
				
	
		return node	
			
			
	
	
	def detCandNumericSplits(self, D, attrIndex, labelVals):
	
		C =[]
	
		S = {}
		uniXs = []
		insInd = 0
	
		numInd = 0
	
		for item in self.train_set.instances:
			if insInd in D:
				# Obtain the i-th feature
				xi = float(item[attrIndex])
				labelInst = item[-1] 
				# Check if this x has been encountered before
				try:
					l = S[xi]
				except KeyError:
					uniXs.append(xi)
					xiInstList = []
					# First label value assumed to be positive
					# Second label value assumed to be negative
					numPos = 0
					numNeg = 0
					xiPosNegList = []
					S[xi] = [xiInstList, xiPosNegList]
					S[xi][0].append(insInd)
					if labelInst == labelVals[0]:
							numPos+= 1
					if labelInst == labelVals[1]:
							numNeg +=1
				
					S[xi][1] = [numPos,numNeg]
		
					numInd += 1
				else:
					S[xi][0].append(insInd)
					if labelInst == labelVals[0]:
							S[xi][1][0]+= 1
					if labelInst == labelVals[1]:
							S[xi][1][1]+=1
		
					numInd += 1
				insInd += 1
			else:
				insInd += 1
				continue
				
		
		uniXs.sort()
		
		
		#	for each pair of adjacent sets sj,sj+1 in sorted S
		for i in range (0,len(uniXs)-1):
		# Atleast one +/-
		# Atleast one -/+ numPos,numNeg
			# obtain pos neg pairs for si
			numPosXi = S[uniXs[i]][1][0]
			numNegXi = S[uniXs[i]][1][1]
			# obtain pos neg pairs for si+1
			numPosXip1 = S[uniXs[i+1]][1][0]
			numNegXip1 = S[uniXs[i+1]][1][1]

			if numPosXi >= 1 and numNegXip1 >= 1:
				midpt = (float(uniXs[i]) + float(uniXs[i+1]))/2.0
				C.append(midpt)
			elif numNegXi >= 1 and numPosXip1 >= 1:
				midpt = (float(uniXs[i]) + float(uniXs[i+1]))/2.0
				C.append(midpt)
			else:
				continue
	
			
		return C
		
	
	

	def determineCandidateSplits(self,D):

		candidateSplits = {}
	
		numAttrs = len(self.train_set.attributes)
		labelDict =  self.train_set.attributes[numAttrs-1]
	
		labelVals =[]
		for k, v in labelDict.iteritems():
			labelVals = v
		# Looping over every attribute
		for i in range(0,len(self.train_set.attributes)-1):
	
			attValDict = self.train_set.attributes[i]
			attrIndex = i
			for k,v in attValDict.iteritems():
				attrName = k
	
			# If attrType is real, set numChildren =2 and assign a list for associated thresholds
			if self.train_set.attributes[i][attrName] == 'real' or self.train_set.attributes[i][attrName] == 'numeric':
				numChildren = 2
				candidateSplits[attrName] = {}				
				thresholds = self.detCandNumericSplits(D, attrIndex, labelVals)
				candidateSplits[attrName][numChildren] = thresholds
			else:
				nomiVals = self.train_set.attributes[i][attrName]
				numChildren = len(nomiVals)
				candidateSplits[attrName] = {}
				candidateSplits[attrName][numChildren] = nomiVals
				
		return candidateSplits
	
	

decisionTree()

	
