import numpy as npy
import matplotlib as mplot
import statistics as stats
file = open("ex1data2.txt","rt")
data = file.read()
data1 = data.split()
x1 = []
x2 = []
y = []
for ele in data1:
    element = ele.split(',')
    x1.append(float(element[0]))
    x2.append(float(element[1]))
    y.append(float(element[2]))

def featureNormalization(x1, x2):
    x1_norm = x1
    x2_norm = x2
    mu1 = sum(x1_norm)/len(x1_norm)
    mu2 = sum(x2_norm)/len(x2_norm)
    sigma1 = stats.stdev(x1)
    sigma2 = stats.stdev(x2)
    i = 0
    for i in range(len(x1_norm)):
        x1_norm[i] = (x1_norm[i] - mu1)/sigma1
        x2_norm[i] = (x2_norm[i] - mu2)/sigma2
    return x1_norm, x2_norm

def polynomial(x1_normI, x2_normI, theta):
    h = theta[0] + theta[1]*x1_normI + theta[2]*x2_normI
    return h

def costFunction(x1_norm, x2_norm, y, theta):
    cost = 0
    i = 0
    for i in range(len(x1_norm)):
        cost += (polynomial(x1_norm[i], x2_norm[i], theta) - y[i])*(polynomial(x1_norm[i], x2_norm[i], theta) - y[i])
    cost = cost/(2*len(x1_norm))
    return cost

def multiVariableGradDescent(x1_norm, x2_norm, y, theta, alpha, maxIter):
    i = 0
    for i in range(maxIter):
        j = 0
        costDer = 0
        theta1 = theta
        for j in range(len(x1_norm)):
                costDer += (polynomial(x1_norm[j], x2_norm[j], theta) - y[j])
        theta1[0] = theta[0] - alpha*(costDer/len(x1_norm))
        costDer = 0
        j = 0
        for j in range(len(x1_norm)):
                costDer += (polynomial(x1_norm[j], x2_norm[j], theta) - y[j])*x1_norm[j]
        theta1[1] = theta[1] - alpha*(costDer/len(x1_norm))
        costDer = 0
        j = 0
        for j in range(len(x1_norm)):
                costDer += (polynomial(x1_norm[j], x2_norm[j], theta) - y[j])*x2_norm[j]
        theta1[2] = theta[2] - alpha*(costDer/len(x1_norm))
        theta = theta1
    return theta

x1_norm, x2_norm = featureNormalization(x1, x2)
theta = []
print("Enter three parameters one by one")
theta.append(float(input()))
theta.append(float(input()))
theta.append(float(input()))
print("Enter learning rate")
alpha = float(input())
print("Enter maximum number of iterations for gradient descent to converge")
maxIter = int(input())
thetaFinal = multiVariableGradDescent(x1_norm, x2_norm, y, theta, alpha, maxIter)
print("Optimized parameters are")
print(thetaFinal)
print("Enter your datapoints one after another")
x1_normI = float(input())
x2_normI = float(input())
hypothesis = polynomial(x1_normI, x2_normI, thetaFinal)
print("The prediction for given datapoints is")
print(hypothesis)


