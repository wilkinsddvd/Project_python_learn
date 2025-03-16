from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Key of iris_datasets:\n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193]+"\n...")

print("Target names:{}".format(iris_dataset['target_names']))

print("Feature names:\n{}".format(iris_dataset['feature_names']))

print("Type of data:{}".format(type(iris_dataset['data'])))

print("Shape of data:{}".format(iris_dataset['data'].shape))

print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))

print("Type of target:{}".format(type(iris_dataset['target'])))

print("Shape of target:{}".format(iris_dataset['target'].shape))

