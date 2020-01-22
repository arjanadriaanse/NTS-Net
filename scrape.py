import numpy as np

log_file = open('log.log')

data = []
for i, line in enumerate(log_file):
    parts = line.split(' ')
    if i % 3 == 0:
        row = []
    elif i % 3 == 1:
        row.append(int(parts[1].split(':')[1])) # epoch
        row.append(float(parts[5])) # training loss
        row.append(float(parts[9])) # training accuracy
    elif i % 3 == 2:
        row.append(float(parts[5])) # testing loss
        row.append(float(parts[9])) # testing accuracy
        data.append(row)

np.savetxt("log.csv", np.array(data), delimiter=",", header="epoch,trainloss,trainacc,testloss,testacc")
