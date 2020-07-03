import numpy as np
import utils
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# setup the randoms tate
RANDOM_STATE = 545510477
#input: X_train, Y_train, X_test
#output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):
	#train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use default params for the classifier
    Logistic_Model = LogisticRegression(random_state=RANDOM_STATE).fit(X_train,Y_train)
    Logistic_pred = Logistic_Model.predict(X_test)
    return Logistic_pred

#input: X_train, Y_train, X_test
#output: Y_pred
def svm_pred(X_train, Y_train, X_test):
	#train a SVM classifier using X_train and Y_train. Use this to predict labels of X_train
	#use default params for the classifier
    SVM_Model = LinearSVC(random_state=RANDOM_STATE).fit(X_train,Y_train)
    SVM_pred = SVM_Model.predict(X_test)
    return SVM_pred

#input: X_train, Y_train, X_test
#output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
	#train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use max_depth as 5
    decisionTree_Model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5).fit(X_train,Y_train)
    decisionTree_pred = decisionTree_Model.predict(X_test)
    return decisionTree_pred


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    return accuracy_score(Y_pred,Y_true),roc_auc_score(Y_pred,Y_true),precision_score(Y_pred,Y_true),recall_score(Y_pred,Y_true),f1_score(Y_pred,Y_true)

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")

def main():
	X_train, Y_train = utils.get_data_from_svmlight("output/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("data/features_svmlight.validate")
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train,X_test),Y_test)
	

if __name__ == "__main__":
	main()
	