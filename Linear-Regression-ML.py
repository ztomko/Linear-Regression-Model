# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 01:24:24 2022

@author: Zack
"""
import matplotlib.pyplot as plt
import numpy as np

#Weight of iterative refinement changes
ALPHA = 0.05  

#Assigns chart colors based on product functionality
def clorz(itWorks):
    colors = []
    for i in itWorks:
        if i == 1:
            colors.append("green")
        else:
            colors.append("red")
    return colors
    
#Sigmoid function
def sig(z):
    return 1.0/(1.0 + np.exp(-z))

#Iterative refinement for standard error (J)
def jiters(cost, itWorks, train, data, w):
    costs = []
    int(input("How many iterations would you like to run?"))
    for i in range(its):
        for j in range((train-1)):
            hyp = sig(np.dot(data, w))
            wtemp = np.dot(data.T, (hyp - itWorks)) / train
            w -= ALPHA * wtemp
        cost = -np.mean(itWorks*(np.log(hyp)) + (1-itWorks)*np.log(1-hyp))
        costs.append(cost)
    plt.title("J vs. Iterations")
    plt.ylabel("J value")
    plt.xlabel("Iterations")
    plt.scatter(range(its), costs)
    plt.savefig("JvsI_LR" + ".png", bbox_inches="tight")
    plt.show()
    
#Creates decision boundary based on test data
def bestFit(data, w, itWorks):
    plt.title("Decision Boundary")
    plt.ylabel("Test 1")
    plt.xlabel("Test 2")
    plt.scatter(data.T[1], data.T[3], c = clorz(itWorks))
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    F = -0.5
#   More accuracy
    F += w[0]
    F += w[1] * X
    F += w[2] * X**2
    F += w[3] * Y
    F += w[4] * Y**2
    F += w[5] * X*Y
#   Less accuracy
#   F += w[6] * 
#   F += w[6] * X**2*Y**2
#   F += w[6] * (X*Y)**2
#   F += w[6] * X**2*Y
#   F += w[6] * X*Y**2
    plt.contour(X,Y,F,[0])
    plt.savefig("Decision_Boundary_LR" + ".png", bbox_inches="tight")
    plt.show()
            
#Gets the values of a confusion matrix using data
def confusion(data, w, itWorks):
    pre = sig(np.dot(data, w))
    count = 0
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in pre:
        test = -1
        if i >= 0.5:
            test = 1
            if test == itWorks[count]:
                TP += 1
            else:
                FP += 1
        else:
            test = 0
            if test == itWorks[count]:
                TN += 1
            else:
                FN += 1
        count += 1
    accuracy = (TN + TP)/(TN + TP + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 / (1/precision + 1/recall)
    print(  "TN: " + str(TN) + 
            "\nTP: " + str(TP) +
            "\nFN: " + str(FN) +
            "\nFP: " + str(FP) +
            "\nAccuracy: " + str(accuracy) +
            "\nPrecision: " + str(precision) +
            "\nRecall: " + str(recall) +
            "\nF1: " + str(F1))

#Reads data in from a file
def readData(size, fin, itWorks, data):
    for i in range(size):
        line = fin.readline().split("\t")
        X = float(line[0])
        Y = float(line[1])
        itWorks[i] = int(line[2])
#       More accuracy
        data[i][0] = 1
        data[i][1] = X
        data[i][2] = X**2
        data[i][3] = Y
        data[i][4] = Y**2
        data[i][5] = X*Y
#       Less accuracy
#       data[i][6] = np.sin(Y*X)
#       data[i][6] = X**2*Y**2
#       data[i][6] = (X*Y)**2
#       data[i][6] = X**2*Y
#       data[i][6] = X*Y**2

#Main function: reads training file, performs logistic regression,
#   and tests the result against the testing file
def main():
    trainFile = input("Enter a file for training: ")
    fin = open(trainFile, "r")
    numS = fin.readline().split("\t")
    train = int(numS[0])
    features = int(9)
    itWorks = np.zeros(train, dtype = "int")
    w = np.zeros(features, dtype = "float")
    cost = 0
    data = np.zeros([train, features], dtype = "float")
    readData(train, fin, itWorks, data)
    fin.close()
    
    jiters(cost, itWorks, train, data, w)
    bestFit(data, w, itWorks)
    
    testFile = input("Enter a file for testing: ")
    fin2 = open(testFile, "r")
    num2 = fin2.readline().split("\t")
    test = int(num2[0])    
    works = np.zeros(test, dtype = "int")
    dat2 = np.zeros([test, features], dtype = "float")
    readData(test, fin2, works, dat2)
    fin2.close()
    print("Training J: " + str(cost))
    hyp = sig(np.dot(dat2, w))
    cost = -np.mean(works*(np.log(hyp)) + (1-works)*np.log(1-hyp))
    print("Testing J: " + str(cost))
    confusion(dat2, w, works)

if __name__ == '__main__':
    main()
