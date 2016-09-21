import pandas
import numpy as np
from scipy import linalg as la #used for eigenvalue decomposition
from sklearn import neighbors #learning algorithm

# how many dimensions should PCA return?
PCA_DIMS = 40
# if PCA should be done including also test data (and verification data)
INCLUDE_TEST_DATA_IN_PCA = False
# if there should be a verification subset from the training data
DO_VERIFICATION = False
# proportion of labeled examples that should be allocated to training data (value >0 and <=1)
TRAIN_SET_RATIO = 0.7


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

def show_image(data, size = 28):
	from PIL import Image
	img = Image.new('L', (size,size), color=0)
	pixels = img.load()
	for i in range(size):
		for j in range(size):
			pixels[i,j] = 255-data[j*size + i]
	img.show()

'''
input treatment
'''
df = pandas.read_csv('train.csv')
train_data = df.as_matrix()[:,1:]
labels = df['label']
#print train_data.shape
#show_image(train_data[3,:])
#print labels[3]
N, p = train_data.shape

# Alternative: split training data into training and verification sets
if DO_VERIFICATION:
	train_sub_set = np.random.binomial(1,TRAIN_SET_RATIO,N)
else:
	train_sub_set = np.ones(N, dtype=np.int)

# Get test data
df = pandas.read_csv('test.csv')
test_data = df.as_matrix()
N_test, p_test = test_data.shape
if INCLUDE_TEST_DATA_IN_PCA:
	data = np.concatenate((train_data,test_data)) # merge them for PCA
	train_sub_set_after_PCA = np.concatenate((train_sub_set,np.zeros(N_test, dtype=np.int)))
else:
	data = train_data[train_sub_set == 1, :]

'''
train phase
'''
# PCA
U, vals, V, datam = PCA(data, PCA_DIMS)

# kNN
clf = neighbors.KNeighborsClassifier()
if INCLUDE_TEST_DATA_IN_PCA:
	clf.fit(U[train_sub_set_after_PCA == 1,:], labels[train_sub_set == 1])
else:
	clf.fit(U, labels[train_sub_set == 1])

'''
verification phase
'''
if DO_VERIFICATION:
	if INCLUDE_TEST_DATA_IN_PCA:
		# U already contains the examples for verification
		verification_prediction = clf.predict(U[:N,:][train_sub_set==0,:])
	else:
		# U does not contain the examples for verification, so we need to apply the same transformation to it
		verification_prediction = clf.predict(np.dot(V.T,(train_data[train_sub_set == 0, :] - datam).T).T)
	print('Validation accuracy: ' + str(np.average(verification_prediction == labels[train_sub_set == 0])))

'''
apply to test data
'''
if INCLUDE_TEST_DATA_IN_PCA:
	prediction = clf.predict(U[N:, :])
else:
	prediction = clf.predict(np.dot(V.T,(test_data - datam).T).T)

'''
output predictions
'''
with open('output.csv', 'w') as outfile:
	outfile.write('ImageId,Label\n')
	for i in xrange(len(test_data)):
		outfile.write(str(i+1) + ',' + str(prediction[i]) + '\n')

