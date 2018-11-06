import numpy as np
import sys

input_data = np.genfromtxt(sys.argv[1], delimiter = ",")
w1_w2_data1 = input_data[:,:2]
w1_w2_data = input_data[:,:2]
print(w1_w2_data)
y_data = input_data[:,2]
weights = [0,0,0]
iterations = 100
final_out = []
alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.0230]
mu = w1_w2_data.mean(axis = 0)
stdev = w1_w2_data.std(axis = 0)
column_shift = np.ones((input_data.shape[0],1))

# Scaling features

for value in w1_w2_data:
    for index in range(len(value)):
        value[index] = (value[index] - mu[index])/stdev[index]

#print(w1_w2_data)

w1_w2_data = np.concatenate((column_shift,w1_w2_data), axis = 1)

def LR(w1_w2_data, y_data, weights, iterations, alpha):
    B = [0, 0, 0]
    l1 = 1


    while iterations != 0:
        n = 0
        for index in range(len(w1_w2_data)):
            new = 0

            for value in range(len(w1_w2_data[index])):

                temp = w1_w2_data[index]
                new += temp[value]*weights[value]

            n += (new - y_data[index])**2

            for k in range(len(B)):
                B[k] += (new - y_data[index])*w1_w2_data[index][k]

        n = n/(2*len(w1_w2_data))

        if n < l1:
            l1 = n
            weights[0] -= alpha*B[0]/(len(w1_w2_data))
            weights[1] -= alpha*B[1]/(len(w1_w2_data))
            weights[2] -= alpha*B[2]/(len(w1_w2_data))

        else:
            break



        iterations = iterations - 1



    return [weights[0],weights[1],weights[2]]


for index in range(len(alpha)):
    iterations = 100
    b_0, b_age, b_weight = LR(w1_w2_data, y_data, weights, iterations, alpha[index])
    weights = [b_0, b_age, b_weight]
    final_out.append([alpha[index], iterations, b_0, b_age, b_weight])


np.savetxt(sys.argv[2], final_out, delimiter = ",")
