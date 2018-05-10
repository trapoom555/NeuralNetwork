import numpy as np 
import random
import matplotlib.pyplot as plt 
TrainingData = []
InputData = []
ListOfGuessedLabel = []
NumberOfTraining = 300
numberOfInput = 100
numberOfTrainingData = 300
lr = 0.1
AccuracyCount = 0
def f(x):
    y = -2*x*x + 0.1
    return y

def graph(formula, x_range , k):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y , k)  


class ClassTrainingData():
    def __init__(self):
        self.x = random.uniform(-1,1)
        self.y = random.uniform(-1,1)
        self.b = 1
        self.Label = 0
    def LabelTrainingData(self):
        if(self.y >= self.x):
            self.Label = 1
        else:
            self.Label = -1


class ClassInputData():
    def __init__(self):
        self.x = random.uniform(-1,1)
        self.y = random.uniform(-1,1)
        self.b = 1
        self.Label = 0
        
class Perceptron():
    def __init__(self):
        self.w = [0.5, 0.5 , 0.5]
    def ActivatedGuess(self,TrainingData,TrainingNum): # loop Through all TrainingData
        k = 0
        n = (TrainingData[TrainingNum].x * self.w[0]) + (TrainingData[TrainingNum].y * self.w[1])+ (TrainingData[TrainingNum].b * self.w[2])
        if (n >= 0) :
            k = 1
        else :
            k = -1
        return k
    def AdjustWeight(self,ActivatedGuess,TrainingData,TrainingNum): #loop number of training outside loop Through all TrainingData 
        Error = TrainingData[TrainingNum].Label - ActivatedGuess[TrainingNum]
        self.w[0] += Error * TrainingData[TrainingNum].x * lr
        self.w[1] += Error * TrainingData[TrainingNum].y * lr
        self.w[2] += Error * lr
brain = Perceptron()
for i in range(numberOfTrainingData):
    TrainingData.append(ClassTrainingData()) # Create all TraningData objects
    TrainingData[i].LabelTrainingData() #Correct Lable

ListOfGuessedLabel = np.zeros(numberOfTrainingData) # Give Label to Guessed Data

for i in range(NumberOfTraining):
    for j in range(numberOfTrainingData):
        ListOfGuessedLabel[j] = brain.ActivatedGuess(TrainingData,j) # Change Label to the more optimal value 
    for j in range(numberOfTrainingData):
        brain.AdjustWeight(ListOfGuessedLabel,TrainingData,j) # calculate Error and adjust weights
#pyplot
for i in range(numberOfTrainingData):   
    if(TrainingData[i].Label == ListOfGuessedLabel[i]):
        plt.plot(TrainingData[i].x,TrainingData[i].y,'go')
        AccuracyCount += 1
    else:
        plt.plot(TrainingData[i].x,TrainingData[i].y,'ro')
plt.title('Mechine Learning Accuracy : %.2f' % ((AccuracyCount / numberOfTrainingData) * 100))
m =  (brain.w[0] / brain.w[1])
b =  (brain.w[2] / brain.w[1])
graph('(m*x*x) + b' ,np.arange(-1,1,0.1),'r')
graph('f(x)',np.arange(-1,1,0.01),'--')
plt.show()