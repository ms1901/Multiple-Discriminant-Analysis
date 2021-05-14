from numpy.linalg import inv,pinv
import numpy as np
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
#Gettting unique targets
valid_classes=np.unique(y_train)
#Getting length of classes
num_valid_classes=len(valid_classes)
#Visualising 5 samples from each class
for j in range(0,10):
    data = np.where(y_train == j)[0]
    data=data[0:5]
    
    fig, ax = plt.subplots(1,5)
    i=0
    for ax in ax.flatten():
        plottable_image = np.reshape(x_train[data[i]], (28, 28))
        ax.imshow(plottable_image, cmap='gray_r')
        
        i=i+1

#Converting data to dictionary
seperate_data_class={}
for x,y in zip(x_train,y_train):
  if y in seperate_data_class:
    seperate_data_class[y].append(x.flatten())
  else:
    seperate_data_class[y]=[x.flatten()]
for i in valid_classes:
  seperate_data_class[i]=np.array(seperate_data_class[i])

#Converting TESTING data to dictionary
seperate_data_class_test={}
for x,y in zip(x_test,y_test):
  if y in seperate_data_class_test:
    seperate_data_class_test[y].append(x.flatten())
  else:
    seperate_data_class_test[y]=[x.flatten()]
for i in valid_classes:
  seperate_data_class_test[i]=np.array(seperate_data_class_test[i])

data_train=[]
output_train=[]
for out,data_i in seperate_data_class.items():
  data_train.extend(data_i)
  output_train.extend(data_i.shape[0] * [out])
output_train=np.array(output_train) #labels
data_train=np.asarray(data_train)   #data

data_test=[]
output_test=[]
for out,data_i in seperate_data_class_test.items():
  data_test.extend(data_i)
  output_test.extend(data_i.shape[0] * [out])
output_test=np.array(output_test) #labels
data_test=np.asarray(data_test)   #data



#Calculating mean of each class:0,1,2,3,.....9
means_each_class={}
for class_i,i in seperate_data_class.items():
  means_each_class[class_i]=np.mean(i,axis=0)
#print(len(means_each_class))-->10

#Calculating within class scatter mattrix
Sw_i = []
for class_i, m in means_each_class.items():

  sub = np.subtract(seperate_data_class[class_i], m)
  Sw_i.append(np.dot(np.transpose(sub), sub))

Sw_i = np.asarray(Sw_i)
Sw = np.sum(Sw_i, axis=0) # shape = (D,D) Sw=S1+S2+S3+....Sc
print(Sw.shape)

#Calculating total number of points for each class and total mean
N_i = {}
sum_of_all_data_points = 0
for class_id, data in seperate_data_class.items():
  N_i[class_id] = data.shape[0]
  sum_of_all_data_points += np.sum(data, axis=0)

total_n = sum(list(N_i.values()))

m_global = sum_of_all_data_points / total_n #global_mean

#Calculating between class scatter
Sb = []
for class_id, mean_class_i in means_each_class.items():
    sub = mean_class_i - m
    Sb.append(np.multiply(N_i[class_id], np.outer(sub, sub.T)))
Sb = np.sum(Sb, axis=0)
print(Sb.shape)

#finding eigen vectors and values
matrix_needed=np.dot(pinv(Sw), Sb)
print(matrix_needed.shape)
eigen_values, eigen_vectors = eig(matrix_needed)
eiglist = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)
W = np.array([eiglist[i][1] for i in range(9)])
W = np.asarray(W).T

#Classifying using QDA
means = {}
covariance = {}
priors = {} 
for class_id, i in seperate_data_class.items():
  proj = np.dot(i, W)
  print("proj_shape"+str(proj.shape))
  means[class_id] = np.mean(proj, axis=0)
  #print("mean_shape"+str(means[class_id]))
  covariance[class_id] = np.cov(proj, rowvar=False)
  priors[class_id] = i.shape[0] / total_n

def QDA_acc(X,y):
    proj = np.dot(X, W)
    gaussian_likelihoods = []
    classes = sorted(list(means.keys()))
    for x in proj:
      row = []
      for c in classes: 
        first_term = (1. / ((2 * np.pi) ** (x.shape[0] / 2.))) * (1 / np.sqrt(np.linalg.det(covariance[c])))
        x_sub = np.subtract(x, means[c])
        g= first_term* np.exp(-np.dot(np.dot(x_sub, inv(covariance[c])), x_sub.T) / 2.)
        res = priors[c] * g
        row.append(res)

      gaussian_likelihoods.append(row)

    gaussian_likelihoods = np.asarray(gaussian_likelihoods)
    
    predictions = np.argmax(gaussian_likelihoods, axis=1)
    print("predictions"+str(predictions))
    return (np.sum(predictions == y) / len(y))*100



acc = QDA_acc(data_train,output_train)
print("Train acc:", acc)

acc = QDA_acc(data_test,output_test)
print("Test accuracy:", acc)

