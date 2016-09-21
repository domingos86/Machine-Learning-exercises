import pandas
import numpy as np
from scipy import linalg as la

def PCA(data, target_dims=2):
	datam = data.mean(axis=0)
	data -= datam
	
	R = np.cov(data, rowvar = False)
	
	evals, evecs = la.eigh(R)
	
	idx = np.argsort(evals)[::-1]
	
	evecs = evecs[:,idx]
	
	evals = evals[idx]
	
	evecs = evecs[:,:target_dims]
	
	return np.dot(evecs.T, data.T).T, evals, evecs, datam

def read_input(filename='train.csv'):
	return pandas.read_csv(filename)

def show_image(data, size = 28):
	from PIL import Image
	img = Image.new('L', (size,size), color=0)
	pixels = img.load()
	for i in range(size):
		for j in range(size):
			pixels[i,j] = 255-data[j*size + i]
	img.show()

'''
train phase
'''
df = read_input()
train_data = df.as_matrix()[:,1:]
labels = df['label']
#print train_data.shape
#show_image(train_data[3,:])
#print labels[3]
N, p = train_data.shape
#train_sub_set = np.random.binomial(1,0.6,N)

df = read_input('test.csv')
test_data = df.as_matrix()
data = np.concatenate((train_data,test_data))

tdims=40

#U, vals, V, datam = PCA(train_data[train_sub_set == 1,:], tdims)
U, vals, V, datam = PCA(data, tdims)
'''
averages = np.empty([10,10])
for i in range(10):
	averages[i,:] = U[labels == i, :].mean(axis=0)
'''


#from sklearn import svm
#clf = svm.SVC()
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()

#clf.fit(U, labels[train_sub_set == 1])
#clf.fit(U[train_sub_set == 1,:], labels[train_sub_set == 1])
clf.fit(U[:N,:], labels)
#clf.fit(train_data[train_sub_set == 1,:], labels[train_sub_set == 1])

#test_set = np.dot(V.T,(train_data[train_sub_set == 0, :] - datam).T).T
#prediction = clf.predict(test_set)
prediction = clf.predict(U[N:, :])
#prediction = clf.predict(train_data[train_sub_set == 0, :])
#print np.average(prediction == labels[train_sub_set == 0])
print 'ImageId,Label'
for i in xrange(len(test_data)):
	print str(i+1) + ',' + str(prediction[i])




#print vals[:10]
'''
import matplotlib.pyplot as plt
plt.scatter(U[:,0], U[:,1])
plt.show()
'''


'''
from numpy.random import randn
data = np.array([randn(8) for k in range(150)])
data[:50, 2:4] += 5
data[50:, 2:5] += 5

from matplotlib.pyplot import subplots, show
trans = PCA(data, 3)[0]
fig, (ax1, ax2) = subplots(1, 2)
ax1.scatter(data[:50, 0], data[:50, 1], c = 'r')
ax1.scatter(data[50:, 0], data[50:, 1], c = 'b')
ax2.scatter(trans[:50, 0], trans[:50, 1], c = 'r')
ax2.scatter(trans[50:, 0], trans[50:, 1], c = 'b')
show()
'''
