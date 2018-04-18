import random
import matplotlib.pyplot as plt 

n = 300
LearningRate = 1.5
correction = 0
percentCorrection = 0

class Data:
    def __init__(self):
        self.x = random.uniform(-1,1)
        self.y = random.uniform(-1,1)
        self.label = 0

class Perceptron:
    def __init__(self) :
        self.w = []
        self.sumweightres = []
        self.guessRes = []
        self.CalculatedRes = []
        for i in range(2):
            self.w.append(random.uniform(-1,1)) #weight in x axis and y axis
    def SumWeight(self,i):
        sum = (TrainingData[i].x * self.w[0]) + (TrainingData[i].y * self.w[1])
        return sum

#activation function
def sign(n): 
    if(n>=0):
        return 1
    else:
        return -1

#adjust weight function
def deltaWx(guessRes,TrainingData,i,lr):
    error = TrainingData[i].label - guessRes[i]
    return error * TrainingData[i].x * lr

def deltaWy(guessRes,TrainingData,i,lr):
    error = TrainingData[i].label - guessRes[i]
    return error * TrainingData[i].y * lr

brain = Perceptron()
TrainingData = []
outputs = []
for i in range(n):
    TrainingData.append(Data())
    if(TrainingData[i].x >= TrainingData[i].y):
        plt.plot(TrainingData[i].x,TrainingData[i].y,'go')
        TrainingData[i].label = -1 # actual result
    else:
        plt.plot(TrainingData[i].x,TrainingData[i].y,'ro')
        TrainingData[i].label = 1 # actual result

for i in range(n):
    brain.sumweightres.append(brain.SumWeight(i))
    brain.guessRes.append(sign(brain.SumWeight(i))) # guess result

# adjust weight
for i in range(n):
    brain.w[0] += deltaWx(brain.guessRes,TrainingData,i,LearningRate)
    brain.w[1] += deltaWy(brain.guessRes,TrainingData,i,LearningRate)
    brain.CalculatedRes.append(sign(brain.SumWeight(i)))

for i in range(n):
    if(TrainingData[i].label == brain.CalculatedRes[i]):
        plt.plot(TrainingData[i].x,TrainingData[i].y,'go')
        correction += 1
    else:
        plt.plot(TrainingData[i].x,TrainingData[i].y,'ro')

percentCorrection = ( correction / n ) * 100
plt.title('Mechine learning accuracy : %.2f' % percentCorrection)
plt.plot([-1,1],[-1,1],linewidth=2,color='b')
plt.show()


