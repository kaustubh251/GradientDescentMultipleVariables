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
        cost += (polynomial(x_norm[i], theta) - y[i])*(polynomial(x_norm[i], theta) - y[i])
    cost = cost/(2*len(x_norm))
    return cost

def multiVariableGradDescent(x_norm, y, theta, alpha, maxIter):
    i = 0
    for i in range(maxIter):
        j = 0
        costDer = 0
        theta1 = theta
        for j in range(len(x_norm)):
                costDer += (polynomial(x_norm[j], theta) - y[j])
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


