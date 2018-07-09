# Check the versions of libraries
print("\n")
print("Versions of Libraries\n")
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

print("\n")
print("Load the Data\n")

print("\nIn this step we are going to load the iris data from CSV file URL.\n")

print("Import Libraries\n")

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



print("\nLoad Dataset\n")

print("We can load the data directly from the UCI Machine Learning repository.\nWe are using pandas to load the data.\n We will also use pandas next to explore the data both with descriptive statistics and data visualization.\nNote that we are specifying the names of each column when loading the data.\n This will help later when we explore the data.\n")

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


print("The dataset should load without incident.\nIf you do have network problems, you can download the iris.data file into your working directory and load it using the same method, \nchanging URL to the local file name.\n")

print("Summarize the Dataset\n")

print("\nDimensions of Dataset\n")

print("We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.")

# shape

print("\nYou should see 150 instances and 5 attributes:\n")
print(dataset.shape)

print("\nPeek at the Data\n")

# head
print(dataset.head(20))

print("\nStatistical Summary\n")

#descriptions
print(dataset.describe())

print("\n Class Distribution\n")

print("\n Lets now take a look at the number of instances(rows) that belong to each class. We can view this as an absolute count")

#Class Distribution 
print("We can see that each class has the same number os instances(50 or 335 of the dataset).\n")
print(dataset.groupby('class').size())

print("\nData Visualization\n")

print("We are going to look at two types of plots:\nUnivariate plots to better understand each attribute.\nMultivariate plots to better understand the relationships between attributes.\n")

print("\nUniversal Plots\n")

#box and whisker plots
dataset.plot(kind='box', subplots='True', layout=(2,2), sharex=False, sharey=False)
plt.show()

print("\nHistogram of each input\n")\

#histograms

dataset.hist()
plt.show()


print("\nMultivariate Plots\n")

#scatter plot matrix

scatter_matrix(dataset)
plt.show()

#creating a validation dataset

print("\n Creating a validation dataset\n")

#split out validation dataset
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size= 0.20
seed = 7
X_train,X_Validation, Y_train, Y_Validation= model_selection.train_test_split(X, Y, test_size = validation_size,random_state=seed)


print("\nTest Harness\n")

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

print("\n Build Models\n")

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
print("\n We can see that KNN has the largest estimated accuracy score")

print("\nSelect Best Model\n")




# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print("Make Predictions")

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
