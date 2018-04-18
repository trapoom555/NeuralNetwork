import numpy as np 
import matplotlib.pyplot as plt 
import random

numberOfPoint = 2
numberOfTraining = 10000
lr = 1
class Point:
    def __init__(self):
        self.x = random.uniform(-1,1)
        self.y = random.uniform(-1,1)

class Perceptron:
    def __init__(self):
        #self.w = [random.uniform(-1,1),random.uniform(-1,1)]
        self.m = 0
        self.b = 0
    def GradientDescent(self,data):
        for i in range(numberOfPoint):
            x = data[i].x
            y = data[i].y
            guess = self.m * x + self.b
            error = y - guess
            self.m = self.m + error * x * lr
            self.b = self.b + error * lr

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    plt.show()

pts = []

for i in range(numberOfPoint):
    pts.append(Point())
    plt.plot(pts[i].x,pts[i].y , 'ro')


brain = Perceptron()

for i in range(numberOfTraining):
    brain.GradientDescent(pts)

graph('brain.m * x + brain.b ' , range(-1,2))
plt.show()
