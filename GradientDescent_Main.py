import numpy as npy
import matplotlib as mplot
import statistics as stats
file = open("ex1data2.txt","rt")
data = file.read()
data1 = data.split()
x = []
y = []
count2 = 0
for ele in data1:
    element = ele.split(',')
    m = len(element)
    count = 0
    element2 = []
    while count<(m-1):
        element2.append(1)
        element2.append(float(element[count]))
        count += 1
    x.insert(count2,element2)   
    y.append(float(element[m-1]))
    count2 += 1

def featureNormalization(x):
    x_norm = x
    m = len(x[0])
    mu = []
    sigma = []
    i = 1
    while i<m:
        mu.append(sum(x_norm[][i])/len(x))
        i += 1
    i = 1
    while i<m:
        sigma.append(stats.stdev(x_norm[][i]))
        i += 1
    i = 1
    while i<m:
        j = 0
        while j<len(x_norm):
            x_norm[i][j] = (x_norm[i][j] - mu[i])/sigma[i]
            j += 1
        i += 1    
    return mu, sigma, x_norm

def polynomial(x_normI, theta):
    h = 0
    count = 0
    for element in theta:
        h += element*x_normI[count]
        count += 1
    return h

def costFunction(x_norm, y, theta):
    cost = 0
    i = 0
    for i in range(len(x_norm)):
        cost += (polynomial(x_norm[i][], theta) - y[i])*(polynomial(x_norm[i][], theta) - y[i])
    cost = cost/(2*len(x_norm))
    return cost

def multiVariableGradDescent(x_norm, y, theta, alpha, maxIter):
    i = 0
    for i in range(maxIter):
        k = 0
        theta1 = theta
        while k<len(theta):
            j = 0
            costDer = 0
            for j in range(len(x_norm)):
                costDer += (polynomial(x_norm[j][], theta) - y[j])*x_norm[j][]
            theta1[k] = theta[k] - alpha*(costDer/len(x_norm))
            k += 1
    return theta

mu, sigma, x_norm = featureNormalization(x)
theta = []
print("Enter the number of parameters")
paramNums = int(input())
print("Enter three parameters one by one")
i = 0
while i<paramNums:
    theta.append(float(input()))
    i += 1
print("Enter learning rate")
alpha = float(input())
print("Enter maximum number of iterations for gradient descent to converge")
maxIter = int(input())
thetaFinal = multiVariableGradDescent(x_norm, y, theta, alpha, maxIter)
print("Optimized parameters are")
print(thetaFinal)
print("Enter your datapoints one after another")
x_normI = [1]
i = 0
while i<(paramNums-1):
    x_normI.append(float(input()))
    i += 1
i = 1
while i<paramNums:
    x_normI[i] = (x_normI[i] - mu[i])/sigma[i]
    i += 1
hypothesis = polynomial(x_normI, thetaFinal)
print("The prediction for given datapoints is")
print(hypothesis)
