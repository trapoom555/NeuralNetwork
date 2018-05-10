from TPxNN import *
brain = NeuralNetwork([2,2,1])
brain.inputData([[1,1],[1,0],[0,1],[0,0]])
brain.targetData([[1],[0],[0],[0]])
brain.train(7000)
brain.Feedforward([1,1])
brain.Feedforward([1,0])
brain.Feedforward([0,1])
brain.Feedforward([0,0])
