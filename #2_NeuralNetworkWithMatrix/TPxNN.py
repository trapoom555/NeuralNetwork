import random
from math import exp
class Matrix:
    def __init__(self,x,y):
        self.matrix = [[random.uniform(-1,1) for i in range(y)] for j in range(x)]
def multiply(m1,m2):
        if(len(m1[0]) == len(m2)):
            c = Matrix(len(m1),len(m2[0]))
            for i in range(len(m1)):
                for k in range(len(m2[0])):
                    sum = 0
                    for j in range(len(m1[0])):
                        sum += m1[i][j]*m2[j][k]
                    c.matrix[i][k] = sum
            return(c.matrix)            
        else:
            print("Error : Can't multiply Matrices")

def smultiply(m1,m2):
    if(len(m1) == len(m2) and len(m1[0]) == len(m2[0])):
        c = Matrix(len(m1),len(m1[0]))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                c.matrix[i][j] = m1[i][j] * m2[i][j]
        return(c.matrix)
    else:
        print("Error : Can't smultiply Matrices")

def scalarmultiply(n,m):
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = m[i][j] * n
    return(m)

def add(m1,m2):
    if(len(m1) == len(m2) and len(m1[0]) == len(m2[0])):
        c = Matrix(len(m1),len(m1[0]))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                c.matrix[i][j] = m1[i][j] + m2[i][j]
        return(c.matrix)
    else:
        print("Error : Can't add Matricies")

def subtract(m1,m2):
    if(len(m1) == len(m2) and len(m1[0]) == len(m2[0])):
        c = Matrix(len(m1),len(m1[0]))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                c.matrix[i][j] = m1[i][j] - m2[i][j]
        return(c.matrix)
    else:
        print("Error : Can't subtract Matricies")

def transpose(m):
    c = Matrix(len(m[0]),len(m))
    for j in range(len(m)):
        for i in range(len(m[0])):
            c.matrix[i][j] = m[j][i]
    return(c.matrix)

def ActivationFunction(m1):
    c = Matrix(len(m1),len(m1[0]))
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            c.matrix[i][j] = 1/(1+exp(-m1[i][j]))
    return(c.matrix)


def dSigmoid(m1):
    c = Matrix(len(m1),len(m1[0]))
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            c.matrix[i][j] = m1[i][j] * ( 1 - m1[i][j])
    return(c.matrix)



class NeuralNetwork:
    def __init__(self,listofarchitecture):
        self.lr = 0.1
        self.achitecture = listofarchitecture
        self.inputTraining = []
        self.targetTraining = []
        self.w = [0 for i in range(len(listofarchitecture) - 1)]
        self.data = [0 for i in range(len(listofarchitecture))]
        self.datab = [0 for i in range(len(listofarchitecture))]
        self.bias = [0 for i in range(len(listofarchitecture) - 1)]
        self.error = [0 for i in range(len(listofarchitecture) - 1)]
        self.deltaw = [0 for i in range(len(listofarchitecture) - 1)]
        self.deltab = [0 for i in range(len(listofarchitecture) - 1)]
        for layer in range(len(self.achitecture)):
            self.data[layer] = Matrix(self.achitecture[layer],1)
            self.datab[layer] = Matrix(self.achitecture[layer],1)
        for layer in range(len(self.achitecture) - 1):
            self.w[layer] = Matrix(self.achitecture[layer+1],self.achitecture[layer])
            self.bias[layer] = Matrix(self.achitecture[layer+1],1)
        for layer in range(len(self.achitecture) - 1):
            self.error[layer] = Matrix(self.achitecture[layer + 1],1)
            self.deltaw[layer] = Matrix(1,1)
            self.deltab[layer] = Matrix(1,1)

    def inputData(self , listOfInputs):
        for j in range(len(listOfInputs)):
            self.inputTraining.append(Matrix(len(listOfInputs[j]),1))
            for i in range(len(listOfInputs[j])):
                self.inputTraining[j].matrix[i][0] = listOfInputs[j][i]

    def targetData(self , listOfTargets):
        for j in range(len(listOfTargets)):
            self.targetTraining.append(Matrix(len(listOfTargets[j]),1))
            for i in range(len(listOfTargets[j])):
                self.targetTraining[j].matrix[i][0] = listOfTargets[j][i]

    def Feedforward(self,inputData):
        MatrixOfInputData = Matrix(self.achitecture[0],1)
        for i in range(self.achitecture[0]):
            MatrixOfInputData.matrix[i][0] = inputData[i]
        self.data[0] = MatrixOfInputData
        for layer in range(len(self.achitecture) - 1):
            self.datab[layer+1].matrix = add(multiply(self.w[layer].matrix,self.data[layer].matrix),self.bias[layer].matrix)
            self.data[layer+1].matrix = ActivationFunction(self.datab[layer + 1].matrix)
        print(self.data[-1].matrix)
    
    def train(self,NumberOfTraining):
        for i in range(NumberOfTraining):
            for numberOfTrainingData in range(len(self.inputTraining)):
                #FeedForward
                self.data[0] = self.inputTraining[numberOfTrainingData]
                for layer in range(len(self.achitecture) - 1):
                    self.datab[layer+1].matrix = add(multiply(self.w[layer].matrix,self.data[layer].matrix),self.bias[layer].matrix)
                    self.data[layer+1].matrix = ActivationFunction(self.datab[layer + 1].matrix)
                #BackPropagation      
                self.error[-1].matrix = subtract(self.targetTraining[numberOfTrainingData].matrix,self.data[-1].matrix)
                for layer in range(len(self.achitecture) - 3,-1,-1):
                    self.error[layer].matrix = multiply(transpose(self.w[layer+1].matrix),self.error[layer+1].matrix)
                for layer in range(len(self.achitecture) - 1):
                    self.deltaw[layer].matrix = multiply(scalarmultiply(self.lr,smultiply(self.error[layer].matrix,dSigmoid(self.data[layer+1].matrix))),transpose(self.data[layer].matrix))
                    self.deltab[layer].matrix = scalarmultiply(self.lr,smultiply(self.error[layer].matrix,dSigmoid(self.data[layer+1].matrix)))
                    self.w[layer].matrix = add(self.w[layer].matrix,self.deltaw[layer].matrix)
                    self.bias[layer].matrix = add(self.bias[layer].matrix,self.deltab[layer].matrix)



