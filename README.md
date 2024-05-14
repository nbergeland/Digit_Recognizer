# Digit_Recognizer
Summary: 
 
This is the code I built for an algorithm trained to recognize images of numerical digits 1-10.  The model I built tested at 99.175% recognition, which was good enough for 378 / 1806 (top 21%) in the corresponding Kaggle competition. 
 	The model was built and trained with a combination of matplotlib, numpy, seaborne, tenorflow, SKlearn, and Keras.  I first read in the testing and training datasets for the various models to work with.  From this point, I began understanding the dataset with some basic EDA.    	To perform the basic analysis, I used a graph within matplotlib to see the occurrence of different numbers.  Is the dataset normally distributed?  After observing the normality (shape) of the dataset, I got to work.  The work involved some simple encoding, then creation of different layers of the CNN model.   
 	The thing I was not prepared for was how time intensive the model was.  Each different visualization model took 45-60 minutes to run through all of the sequences of testing.  I found it fascinating to watch the model simulate through “epochs” of data and continually improve itself with no guidance provided by me.   
 	The result was just as impressive.  With no structure provided by me, the model was recognizing nearly every number it was presented with.  One could argue the 99 + % success rate may be below a human, which would likely get all of the numbers.  However, with no prior learning and in such a short period, it is truly shocking the success the model was able to develop given the timeframe. 
 	
 
Appendix: 
 
 
1	0 0 	0 	0 	0 	0 	0 	0 	0 	0 	...
0	0 
2	1 0 	0 	0 	0 	0 	0 	0 	0 	0 	...
0	0 
3	4 0 	0 	0 	0 	0 	0 	0 	0 	0 	...
0	0 
4	0 0 	0 	0 	0 	0 	0 	0 	0 	0 	...
0	0 
5	rows × 785 columns 
 
 
 
Out[7]: (42000, 10) 
  
0	1  
1	0  
2	1  
3	4  
4	0  
Name: label, dtype: int64  
Out[8]: array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],        [1., 0., 0., 0., 0., 0., 
0., 0., 0., 0.],  
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  
 
 _________________________________________________________________  
Layer (type)                 Output Shape              Param #    
============================================================
===== input_1 (InputLayer)         (None, 28, 28, 1)         0          
____________________________________________________________
_____ layer_conv1 (Conv2D)         (None, 28, 28, 64)        640        
____________________________________________________________
_____ batch_normalization_1 (Batch (None, 28, 28, 64)        256        
____________________________________________________________
_____ activation_1 (Activation)    (None, 28, 28, 64)        0          
____________________________________________________________
_____ maxPool1 (MaxPooling2D)      (None, 14, 14, 64)        0          
____________________________________________________________
_____ layer_conv2 (Conv2D)         (None, 14, 14, 32)        18464      
____________________________________________________________
_____ batch_normalization_2 (Batch (None, 14, 14, 32)        128        
____________________________________________________________
_____ activation_2 (Activation)    (None, 14, 14, 32)        0          
____________________________________________________________
_____ maxPool2 (MaxPooling2D)      (None, 7, 7, 32)          0          
____________________________________________________________
_____ conv3 (Conv2D)               (None, 7, 7, 32)          9248       
____________________________________________________________
_____ batch_normalization_3 (Batch (None, 7, 7, 32)          128        
____________________________________________________________
_____ activation_3 (Activation)    (None, 7, 7, 32)          0          
____________________________________________________________
_____ maxPool3 (MaxPooling2D)      (None, 3, 3, 32)          0          
____________________________________________________________
_____ flatten_1 (Flatten)          (None, 288)               0          
____________________________________________________________
_____ fc0 (Dense)                  (None, 64)                18496      
____________________________________________________________
_____ dropout_1 (Dropout)          (None, 64)                0          
____________________________________________________________
_____ fc1 (Dense)                  (None, 32)                2080       
____________________________________________________________
_____ dropout_2 (Dropout)          (None, 32)                0          
____________________________________________________________
_____ fc2 (Dense)                  (None, 10)                330         
============================================================ =====  
Total params: 49,770  
Trainable params: 49,514  
Non-trainable params: 256  
_________________________________________________________________  
In [13]: # Adam optimizer conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metr ics=['accuracy'])  conv_model.fit(X_train, y_train, epochs=10, batch_size=100, validation_d ata=(X_cv,y_cv))  
Train on 37800 samples, validate on 4200 samples  
Epoch 1/10  
37800/37800 [==============================] - 144s 4ms/step - loss: 0. 
4887 - accuracy: 0.8468 - val_loss: 0.6940 - val_accuracy: 0.8188  
Epoch 2/10  
37800/37800 [==============================] - 138s 4ms/step - loss: 0. 
1349 - accuracy: 0.9610 - val_loss: 0.0763 - val_accuracy: 0.9776  
Epoch 3/10  
37800/37800 [==============================] - 138s 4ms/step - loss: 0. 
0911 - accuracy: 0.9744 - val_loss: 0.0541 - val_accuracy: 0.9857  
Epoch 4/10  
37800/37800 [==============================] - 138s 4ms/step - loss: 0. 
0760 - accuracy: 0.9791 - val_loss: 0.0640 - val_accuracy: 0.9840  
Epoch 5/10  
37800/37800 [==============================] - 138s 4ms/step - loss: 0. 
0629 - accuracy: 0.9824 - val_loss: 0.0658 - val_accuracy: 0.9795  
Epoch 6/10  
37800/37800 [==============================] - 142s 4ms/step - loss: 0. 
0566 - accuracy: 0.9845 - val_loss: 0.0549 - val_accuracy: 0.9838  
Epoch 7/10  
37800/37800 [==============================] - 139s 4ms/step - loss: 0. 
0471 - accuracy: 0.9872 - val_loss: 0.0378 - val_accuracy: 0.9902  
Epoch 8/10  
37800/37800 [==============================] - 139s 4ms/step - loss: 0. 
0460 - accuracy: 0.9872 - val_loss: 0.0558 - val_accuracy: 0.9864  
Epoch 9/10  
37800/37800 [==============================] - 139s 4ms/step - loss: 0. 
0363 - accuracy: 0.9893 - val_loss: 0.0427 - val_accuracy: 0.9888  
Epoch 10/10  
37800/37800 [==============================] - 140s 4ms/step - loss: 0. 
0366 - accuracy: 0.9899 - val_loss: 0.0680 - val_accuracy: 0.9850  
Out[13]: <keras.callbacks.callbacks.History at 0x7f8bee1d2850> 
In [14]: # SGD optimizer sgd = SGD(lr=0.0005, momentum=0.5, decay=0.0, nesterov=False)  conv_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics =['accuracy'])  conv_model.fit(X_train, y_train, epochs=30, validation_data=(X_cv, y_cv 
	))  	 
Train on 37800 samples, validate on 4200 samples  
Epoch 1/30  
37800/37800 [==============================] - 173s 5ms/step - loss: 0. 
0288 - accuracy: 0.9922 - val_loss: 0.0345 - val_accuracy: 0.9929  
Epoch 2/30  
37800/37800 [==============================] - 170s 5ms/step - loss: 0. 
0237 - accuracy: 0.9937 - val_loss: 0.0337 - val_accuracy: 0.9929  
Epoch 3/30  
37800/37800 [==============================] - 172s 5ms/step - loss: 0. 
0208 - accuracy: 0.9943 - val_loss: 0.0327 - val_accuracy: 0.9929  
Epoch 4/30  
37800/37800 [==============================] - 1599s 42ms/step - loss: 
0.0188 - accuracy: 0.9949 - val_loss: 0.0327 - val_accuracy: 0.9938  
Epoch 5/30  
37800/37800 [==============================] - 1359s 36ms/step - loss: 
0.0192 - accuracy: 0.9947 - val_loss: 0.0330 - val_accuracy: 0.9926  
Epoch 6/30  
37800/37800 [==============================] - 322s 9ms/step - loss: 0. 
0178 - accuracy: 0.9951 - val_loss: 0.0327 - val_accuracy: 0.9929  
Epoch 7/30  
37800/37800 [==============================] - 1389s 37ms/step - loss: 
0.0183 - accuracy: 0.9947 - val_loss: 0.0323 - val_accuracy: 0.9933  
Epoch 8/30  
37800/37800 [==============================] - 2006s 53ms/step - loss: 
0.0186 - accuracy: 0.9948 - val_loss: 0.0320 - val_accuracy: 0.9933  
Epoch 9/30  
37800/37800 [==============================] - 1214s 32ms/step - loss: 
0.0162 - accuracy: 0.9960 - val_loss: 0.0321 - val_accuracy: 0.9938  
Epoch 10/30  
37800/37800 [==============================] - 177s 5ms/step - loss: 0. 
0172 - accuracy: 0.9952 - val_loss: 0.0324 - val_accuracy: 0.9938  
Epoch 11/30  
37800/37800 [==============================] - 174s 5ms/step - loss: 0. 
0166 - accuracy: 0.9957 - val_loss: 0.0329 - val_accuracy: 0.9940  
Epoch 12/30  
37800/37800 [==============================] - 172s 5ms/step - loss: 0. 
0170 - accuracy: 0.9952 - val_loss: 0.0325 - val_accuracy: 0.9940  
Epoch 13/30  
37800/37800 [==============================] - 568s 15ms/step - loss:  
0.0169 - accuracy: 0.9953 - val_loss: 0.0319 - val_accuracy: 0.9943  
Epoch 14/30  
37800/37800 [==============================] - 180s 5ms/step - loss: 0. 
0158 - accuracy: 0.9958 - val_loss: 0.0321 - val_accuracy: 0.9938  Epoch 15/30  
37800/37800 [==============================] - 179s 5ms/step - loss: 0. 
0156 - accuracy: 0.9959 - val_loss: 0.0321 - val_accuracy: 0.9938  
Epoch 16/30  
37800/37800 [==============================] - 179s 5ms/step - loss: 0. 
0148 - accuracy: 0.9963 - val_loss: 0.0324 - val_accuracy: 0.9936  
Epoch 17/30  
37800/37800 [==============================] - 181s 5ms/step - loss: 0. 
0163 - accuracy: 0.9953 - val_loss: 0.0328 - val_accuracy: 0.9938  
Epoch 18/30  
37800/37800 [==============================] - 180s 5ms/step - loss: 0. 
0153 - accuracy: 0.9958 - val_loss: 0.0328 - val_accuracy: 0.9936  
Epoch 19/30  
37800/37800 [==============================] - 184s 5ms/step - loss: 0. 0159 
- accuracy: 0.9958 - val_loss: 0.0331 - val_accuracy: 0.9938  
Epoch 20/30  
37800/37800 [==============================] - 181s 5ms/step - loss: 0. 
0137 - accuracy: 0.9963 - val_loss: 0.0329 - val_accuracy: 0.9938  
Epoch 21/30  
37800/37800 [==============================] - 180s 5ms/step - loss: 0. 
0153 - accuracy: 0.9959 - val_loss: 0.0325 - val_accuracy: 0.9936  
Epoch 22/30  
37800/37800 [==============================] - 180s 5ms/step - loss: 0. 
0154 - accuracy: 0.9958 - val_loss: 0.0325 - val_accuracy: 0.9933  
Epoch 23/30  
37800/37800 [==============================] - 182s 5ms/step - loss: 0. 
0151 - accuracy: 0.9957 - val_loss: 0.0324 - val_accuracy: 0.9936  
Epoch 24/30  
37800/37800 [==============================] - 182s 5ms/step - loss: 0. 
0152 - accuracy: 0.9958 - val_loss: 0.0328 - val_accuracy: 0.9938  
Epoch 25/30  
37800/37800 [==============================] - 183s 5ms/step - loss: 0. 
0134 - accuracy: 0.9966 - val_loss: 0.0323 - val_accuracy: 0.9940  
Epoch 26/30  
37800/37800 [==============================] - 182s 5ms/step - loss: 0. 
0145 - accuracy: 0.9959 - val_loss: 0.0323 - val_accuracy: 0.9943  
Epoch 27/30  
37800/37800 [==============================] - 187s 5ms/step - loss: 0. 
0130 - accuracy: 0.9965 - val_loss: 0.0326 - val_accuracy: 0.9938  
Epoch 28/30  
37800/37800 [==============================] - 182s 5ms/step - loss: 0. 
0137 - accuracy: 0.9960 - val_loss: 0.0324 - val_accuracy: 0.9940  
Epoch 29/30  
37800/37800 [==============================] - 183s 5ms/step - loss: 0. 0139 - accuracy: 0.9961 - val_loss: 0.0329 - val_accuracy: 0.9936  
Epoch 30/30  
37800/37800 [==============================] - 184s 5ms/step - loss: 0. 
0151 - accuracy: 0.9958 - val_loss: 0.0326 - val_accuracy: 0.9933  Out[14]: <keras.callbacks.callbacks.History at 0x7f8bf7b7c350> 
In [ ]:  
 
 
 
 
#confusion matrix 
In [4]: 
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) import os 
In [5]: 
df_train = pd.read_csv("train2.csv") df_test = pd.read_csv("test2.csv") 
In [6]: 
df_train.shape 
Out[6]: 
(42000, 785) 
In [7]: 
y = df_train['label'] 
In [8]: 
df = df_train.drop(['label'], axis=1) df.head() 
Out[8]: 
pi pi pi pi pi pi pi pi pi pi pi pi pi pi pi pi pi pi pi pi  x x x x x x x x x x . xel xel xel xel xel xel xel xel xel xel
. el	el	el	el	el	el	el	el	el	el	77	77	77	77	77	77	78	78	78	78
. 
	0 	1 	2 	3 	4 	5 	6 	7 	8 	9 	4 	5 	6 	7 	8 	9 	0 	1 	2 	3 
0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
pi	pi x el
1 	pi x el
2 	pi x el
3 	pi x el
4 	pi x el
5 	pi x el
6 	pi x el
7 	pi x el
8 	pi x el
9 	.
.
. 	pi xel
77
4 	pi xel
77
5 	pi xel
77
6 	pi xel
77
7 	pi xel
77
8 	pi xel
77
9 	pi xel
78
0 	pi xel
78
1 	pi xel
78
2 	pi xel
78
3 
 	x el
0 																				
1 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	. .	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
. 
2 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
.
3 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
. 
4 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
5 rows × 784 columns 
In [9]: 
# To plot pretty figures %matplotlib inline import matplotlib 
import matplotlib.pyplot as plt plt.rcParams['axes.labelsize'] = 14 plt.rcParams['xtick.labelsize'] = 12 plt.rcParams['ytick.labelsize'] = 12 
In [10]: 
digit_to_predict_raw = np.array(df.iloc[5000,:]) digit_to_predict  = np.array(digit_to_predict_raw).reshape(28,28) digit_to_predict.shape 
Out[10]: 
(28, 28) 
In [11]: 
plt.imshow(digit_to_predict,cmap = matplotlib.cm.binary) plt.show() 
  
In [12]: 
y[5000] 
Out[12]: 
8 
In [13]: 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, ra ndom_state=42) 
In [14]: 
y_train_8 = np.array(y_train == 8) y_test_8 = np.array(y_test == 8) 
In [16]: 
from sklearn.linear_model import SGDClassifier 
 sgd_clf = SGDClassifier(max_iter=5, random_state=42) sgd_clf.fit(X_train, y_train_8) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
Out[16]: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None,               early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True
,               l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=
5,               n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,               random_state=42, shuffle=True, tol=0.001, validation_fraction=0
.1, 
              verbose=0, warm_start=False) 
In [17]: 
pred = sgd_clf.predict(X_test) pred 
Out[17]: 
array([ True, False, False, ..., False, False,  True]) 
In [18]: 
y_test[:4] 
Out[18]: 
5457     8 
38509    1 
25536    9 
31803    9 
Name: label, dtype: int64 
In [19]: 
from sklearn.model_selection import cross_val_score cross_val_score(sgd_clf, X_train, y_train_8, cv=5, scoring="accuracy") /Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
Out[19]: 
array([0.93691549, 0.93673469, 0.93843537, 0.89863946, 0.89743154]) 
In [20]: import collections collections.Counter(y_train_8) 
Out[20]: 
Counter({False: 26546, True: 2854}) 
In [21]: 
num_of_8_not_occur = collections.Counter(y_train_8)[0] print("The accuracy of model if we predict there are NO 8 present in the data set :", 
      num_of_8_not_occur/len(y_train_8)) 
The accuracy of model if we predict there are NO 8 present in the dataset : 0 .9029251700680272 
In [22]: 
#confusion time! from sklearn.model_selection import cross_val_predict from sklearn.metrics import confusion_matrix y_train_8_pred = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3) confusion_matrix(y_train_8, y_train_8_pred) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
Out[22]: 
array([[25244,  1302], 
       [  932,  1922]]) 
In [23]: 
from sklearn import metrics def get_metrics(true_labels, predicted_labels): 
         print('Accuracy:', np.round(                         metrics.accuracy_score(true_labels,                                                 predicted_labels), 
                        4))     print('Precision:', np.round(                         metrics.precision_score(true_labels,                                                 predicted_labels,                                                average='weighted'), 
                        4)) 
    print('Recall:', np.round(                         metrics.recall_score(true_labels,                                                 predicted_labels,                                                average='weighted'), 
                        4))     print('F1 Score:', np.round(                             metrics.f1_score(true_labels,                                                 predicted_labels,                                                average='weighted'),                         4)) 
In [24]: 
get_metrics(y_train_8, y_train_8_pred) 
Accuracy: 0.924 
Precision: 0.9286 
Recall: 0.924 
F1 Score: 0.9261 
In [25]: 
def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]): 
         total_classes = len(classes)     level_labels = [total_classes*[0], list(range(total_classes))] 
     cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels
,                                    labels=classes)     cm_frame = pd.DataFrame(data=cm,                              columns=pd.MultiIndex(levels=[['Predicted:'], cla sses],                                                    labels=level_labels),                              index=pd.MultiIndex(levels=[['Actual:'], classes]
,                                                  labels=level_labels))      print(cm_frame)  
In [26]: 
display_confusion_matrix(y_train_8, y_train_8_pred)           Predicted:        
                   1      0 
Actual: 1       1922    932 
        0       1302  25244 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/ipykernel_ launcher.py:10: FutureWarning: the 'labels' keyword is deprecated, use 'codes
' instead 
  # Remove the CWD from sys.path while we load stuff. /Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/ipykernel_ launcher.py:12: FutureWarning: the 'labels' keyword is deprecated, use 'codes
' instead   if sys.path[0] == '': 
In [27]: 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3,                              method="decision_function") 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/li near_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. 
  ConvergenceWarning) 
In [28]: 
from sklearn.metrics import precision_recall_curve 
 
precisions, recalls, thresholds = precision_recall_curve(y_train_8, y_scores) 
In [29]: 
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):     plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth
=2)     plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)     plt.xlabel("Threshold", fontsize=16)     plt.legend(loc="upper left", fontsize=16)     plt.ylim([0, 1]) 
 plt.figure(figsize=(8, 4)) plot_precision_recall_vs_threshold(precisions, recalls, thresholds) plt.xlim([-1700000, 1700000]) plt.show() 
  
In [30]: 
y_train_prec_90 = (y_scores > 700000) 
In [31]: 
from sklearn.metrics import precision_score, recall_score precision_score(y_train_8, y_train_prec_90) 
Out[31]: 
0.9741176470588235 
In [32]: 
recall_score(y_train_8, y_train_prec_90) 
Out[32]: 
0.14505956552207427 
In [33]: 
#ROC from sklearn.metrics import roc_auc_score, roc_curve fpr, tpr, thresholds = roc_curve(y_train_8, y_scores) 
In [34]: 
def plot_roc_curve(fpr,tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label = label)     plt.plot([0,1], [0,1],'k--')     plt.axis([0,1,0,1])     plt.xlabel('False Positive Rate')     plt.ylabel('True Positive Rate') 
     plot_roc_curve(fpr,tpr) 
 
print("The AUC score is :", roc_auc_score(y_train_8, y_scores)) 
The AUC score is : 0.9098989016751398 
  
In [35]: 
from sklearn.ensemble import RandomForestClassifier forest_clf = RandomForestClassifier(n_jobs=-1) y_forest_pred = cross_val_predict(forest_clf, X_train, y_train_8, cv=3, metho d='predict_proba') y_forest_pred_f = cross_val_predict(forest_clf, X_train, y_train_8, cv=3) /Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/en semble/forest.py:245: FutureWarning: The default value of n_estimators will c hange from 10 in version 0.20 to 100 in 0.22.   "10 in version 0.20 to 100 in 0.22.", FutureWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/en semble/forest.py:245: FutureWarning: The default value of n_estimators will c hange from 10 in version 0.20 to 100 in 0.22.   "10 in version 0.20 to 100 in 0.22.", FutureWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/en semble/forest.py:245: FutureWarning: The default value of n_estimators will c hange from 10 in version 0.20 to 100 in 0.22.   "10 in version 0.20 to 100 in 0.22.", FutureWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/en semble/forest.py:245: FutureWarning: The default value of n_estimators will c hange from 10 in version 0.20 to 100 in 0.22.   "10 in version 0.20 to 100 in 0.22.", FutureWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/en semble/forest.py:245: FutureWarning: The default value of n_estimators will c hange from 10 in version 0.20 to 100 in 0.22.   "10 in version 0.20 to 100 in 0.22.", FutureWarning) 
/Users/nicholasbergeland/opt/anaconda3/lib/python3.7/site-packages/sklearn/en semble/forest.py:245: FutureWarning: The default value of n_estimators will c hange from 10 in version 0.20 to 100 in 0.22. 
  "10 in version 0.20 to 100 in 0.22.", FutureWarning) 
In [36]: 
y_scores_forest= y_forest_pred[:,1] 
fpr_f, tpr_f, thresholds_f = roc_curve(y_train_8, y_scores_forest) 
In [37]: 
plt.plot(fpr, tpr, "b:", label="SGD") 
plot_roc_curve(fpr_f, tpr_f, "Random Forest") plt.legend(loc="lower right") plt.show() 
 print("Classification metrics for SGD :") print("The AUC score for SGD is :", roc_auc_score(y_train_8, y_scores)) get_metrics(y_train_8, y_train_8_pred) print("\nClassification metrics for RandomForest :") print("The AUC score is RandomForest is :", roc_auc_score(y_train_8, y_scores
_forest)) 
get_metrics(y_train_8, y_forest_pred_f) 
  
Classification metrics for SGD : 
The AUC score for SGD is : 0.9098989016751398 
Accuracy: 0.924 
Precision: 0.9286 
Recall: 0.924 
F1 Score: 0.9261 
 
Classification metrics for RandomForest : 
The AUC score is RandomForest is : 0.9850807956106499 
Accuracy: 0.9687 
Precision: 0.9689 
Recall: 0.9687 
F1 Score: 0.9663 
In [39]: 
result = pd.DataFrame(df_test) result 
Out[39]: 
pi  x el
0 	pi x el
1 	pi x el
2 	pi x el
3 	pi x el
4 	pi x el
5 	pi x el
6 	pi x el
7 	pi x el
8 	pi x el
9 	.
.
. 	pi xel
77
4 	pi xel
77
5 	pi xel
77
6 	pi xel
77
7 	pi xel
77
8 	pi xel
77
9 	pi xel
78
0 	pi xel
78
1 	pi xel
78
2 	pi xel
78
3 
0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
.
	1 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
. 
2 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
.
	3 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
. 
4 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
.
	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	.	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 
. 
2
7
9
9
5 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
2
	7	.
	9	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
	9	. 
6 
2
7
9
9
7 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
 
2	pi x el
0 	pi x el
1 	pi x el
2 	pi x el
3 	pi x el
4 	pi x el
5 	pi x el
6 	pi x el
7 	pi x el
8 	pi x el
9 	.
.
. 	pi xel
77
4 	pi xel
77
5 	pi xel
77
6 	pi xel
77
7 	pi xel
77
8 	pi xel
77
9 	pi xel
78
0 	pi xel
78
1 	pi xel
78
2 	pi xel
78
3 
7
9	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	. .	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
	9	. 
8 
2
7
9
9
9 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 	.
.
. 	0 	0 	0 	0 	0 	0 	0 	0 	0 	0 
28000 rows × 784 columns 
In [40]: 
 result.to_csv("confusion_output.csv") 
In [ ]: 
  
 
Works Cited: 
 
sriram2397. “Digit-Recognizer-Kaggle/digit_recognizer.Ipynb at Master · SRIRAM2397/Digit-RecognizerKaggle.” GitHub. Accessed February 6, 2022. https://github.com/sriram2397/digit-recognizerkaggle/blob/master/Digit_ Recognizer.ipynb.   
 
![image](https://github.com/nbergeland/Digit_Recognizer/assets/55772476/f01ab802-a84b-4b25-926a-d7ef2dc24071)


Notebook with machine learning code for Kaggle digit recognizer competition which was able to predict images of digits 0-9 with 99.175% accuracy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
train = pd.read_csv("train2.csv")
test = pd.read_csv("test2.csv")
train.head()
label	pixel0	pixel1	pixel2	pixel3	pixel4	pixel5	pixel6	pixel7	pixel8	...	pixel774	pixel775	pixel776	pixel777	pixel778	pixel779	pixel780	pixel781	pixel782	pixel783
0	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 785 columns

y_train = train['label'].astype('float32')
X_train = train.drop(['label'], axis=1).astype('int32')
X_test = test.astype('float32')
X_train.shape, y_train.shape, X_test.shape
((42000, 784), (42000,), (28000, 784))
sns.countplot(x='label', data=train);

# Data normalization
X_train = X_train/255
X_test = X_test/255
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
X_train.shape, X_test.shape
((42000, 28, 28, 1), (28000, 28, 28, 1))
# one-hot encoding
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
y_train.shape
Using TensorFlow backend.
(42000, 10)
print(train['label'].head())
y_train[0:5,:]
0    1
1    0
2    1
3    4
4    0
Name: label, dtype: int64
array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
plt.imshow(X_train[1][:,:,0])
plt.title(y_train[1].argmax());

from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from keras import backend as K
# Building a CNN model
input_shape = (28,28,1)
X_input = Input(input_shape)

# layer 1
x = Conv2D(64,(3,3),strides=(1,1),name='layer_conv1',padding='same')(X_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool1')(x)
# layer 2
x = Conv2D(32,(3,3),strides=(1,1),name='layer_conv2',padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2),name='maxPool2')(x)
# layer 3
x = Conv2D(32,(3,3),strides=(1,1),name='conv3',padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), name='maxPool3')(x)
# fc
x = Flatten()(x)
x = Dense(64,activation ='relu',name='fc0')(x)
x = Dropout(0.25)(x)
x = Dense(32,activation ='relu',name='fc1')(x)
x = Dropout(0.25)(x)
x = Dense(10,activation ='softmax',name='fc2')(x)

conv_model = Model(inputs=X_input, outputs=x, name='Predict')
conv_model.summary()
Model: "Predict"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 28, 28, 1)         0         
_________________________________________________________________
layer_conv1 (Conv2D)         (None, 28, 28, 64)        640       
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
activation_1 (Activation)    (None, 28, 28, 64)        0         
_________________________________________________________________
maxPool1 (MaxPooling2D)      (None, 14, 14, 64)        0         
_________________________________________________________________
layer_conv2 (Conv2D)         (None, 14, 14, 32)        18464     
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
activation_2 (Activation)    (None, 14, 14, 32)        0         
_________________________________________________________________
maxPool2 (MaxPooling2D)      (None, 7, 7, 32)          0         
_________________________________________________________________
conv3 (Conv2D)               (None, 7, 7, 32)          9248      
_________________________________________________________________
batch_normalization_3 (Batch (None, 7, 7, 32)          128       
_________________________________________________________________
activation_3 (Activation)    (None, 7, 7, 32)          0         
_________________________________________________________________
maxPool3 (MaxPooling2D)      (None, 3, 3, 32)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 288)               0         
_________________________________________________________________
fc0 (Dense)                  (None, 64)                18496     
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
fc1 (Dense)                  (None, 32)                2080      
_________________________________________________________________
dropout_2 (Dropout)          (None, 32)                0         
_________________________________________________________________
fc2 (Dense)                  (None, 10)                330       
=================================================================
Total params: 49,770
Trainable params: 49,514
Non-trainable params: 256
_________________________________________________________________
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

tree_set = df.copy()
target = tree_set.iloc[:,0]
tree_set_X = tree_set.iloc[:,1:] 

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(tree_set_X, target)
clf.score(tree_set_X, target)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-14-c6a67f414158> in <module>
      2 from sklearn import tree
      3 
----> 4 tree_set = df.copy()
      5 target = tree_set.iloc[:,0]
      6 tree_set_X = tree_set.iloc[:,1:]

NameError: name 'df' is not defined
# Adam optimizer
conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_cv,y_cv))
Train on 37800 samples, validate on 4200 samples
Epoch 1/10
37800/37800 [==============================] - 144s 4ms/step - loss: 0.4887 - accuracy: 0.8468 - val_loss: 0.6940 - val_accuracy: 0.8188
Epoch 2/10
37800/37800 [==============================] - 138s 4ms/step - loss: 0.1349 - accuracy: 0.9610 - val_loss: 0.0763 - val_accuracy: 0.9776
Epoch 3/10
37800/37800 [==============================] - 138s 4ms/step - loss: 0.0911 - accuracy: 0.9744 - val_loss: 0.0541 - val_accuracy: 0.9857
Epoch 4/10
37800/37800 [==============================] - 138s 4ms/step - loss: 0.0760 - accuracy: 0.9791 - val_loss: 0.0640 - val_accuracy: 0.9840
Epoch 5/10
37800/37800 [==============================] - 138s 4ms/step - loss: 0.0629 - accuracy: 0.9824 - val_loss: 0.0658 - val_accuracy: 0.9795
Epoch 6/10
37800/37800 [==============================] - 142s 4ms/step - loss: 0.0566 - accuracy: 0.9845 - val_loss: 0.0549 - val_accuracy: 0.9838
Epoch 7/10
37800/37800 [==============================] - 139s 4ms/step - loss: 0.0471 - accuracy: 0.9872 - val_loss: 0.0378 - val_accuracy: 0.9902
Epoch 8/10
37800/37800 [==============================] - 139s 4ms/step - loss: 0.0460 - accuracy: 0.9872 - val_loss: 0.0558 - val_accuracy: 0.9864
Epoch 9/10
37800/37800 [==============================] - 139s 4ms/step - loss: 0.0363 - accuracy: 0.9893 - val_loss: 0.0427 - val_accuracy: 0.9888
Epoch 10/10
37800/37800 [==============================] - 140s 4ms/step - loss: 0.0366 - accuracy: 0.9899 - val_loss: 0.0680 - val_accuracy: 0.9850
<keras.callbacks.callbacks.History at 0x7f8bee1d2850>
# SGD optimizer
sgd = SGD(lr=0.0005, momentum=0.5, decay=0.0, nesterov=False) 
conv_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
conv_model.fit(X_train, y_train, epochs=30, validation_data=(X_cv, y_cv))
Train on 37800 samples, validate on 4200 samples
Epoch 1/30
37800/37800 [==============================] - 173s 5ms/step - loss: 0.0288 - accuracy: 0.9922 - val_loss: 0.0345 - val_accuracy: 0.9929
Epoch 2/30
37800/37800 [==============================] - 170s 5ms/step - loss: 0.0237 - accuracy: 0.9937 - val_loss: 0.0337 - val_accuracy: 0.9929
Epoch 3/30
37800/37800 [==============================] - 172s 5ms/step - loss: 0.0208 - accuracy: 0.9943 - val_loss: 0.0327 - val_accuracy: 0.9929
Epoch 4/30
37800/37800 [==============================] - 1599s 42ms/step - loss: 0.0188 - accuracy: 0.9949 - val_loss: 0.0327 - val_accuracy: 0.9938
Epoch 5/30
37800/37800 [==============================] - 1359s 36ms/step - loss: 0.0192 - accuracy: 0.9947 - val_loss: 0.0330 - val_accuracy: 0.9926
Epoch 6/30
37800/37800 [==============================] - 322s 9ms/step - loss: 0.0178 - accuracy: 0.9951 - val_loss: 0.0327 - val_accuracy: 0.9929
Epoch 7/30
37800/37800 [==============================] - 1389s 37ms/step - loss: 0.0183 - accuracy: 0.9947 - val_loss: 0.0323 - val_accuracy: 0.9933
Epoch 8/30
37800/37800 [==============================] - 2006s 53ms/step - loss: 0.0186 - accuracy: 0.9948 - val_loss: 0.0320 - val_accuracy: 0.9933
Epoch 9/30
37800/37800 [==============================] - 1214s 32ms/step - loss: 0.0162 - accuracy: 0.9960 - val_loss: 0.0321 - val_accuracy: 0.9938
Epoch 10/30
37800/37800 [==============================] - 177s 5ms/step - loss: 0.0172 - accuracy: 0.9952 - val_loss: 0.0324 - val_accuracy: 0.9938
Epoch 11/30
37800/37800 [==============================] - 174s 5ms/step - loss: 0.0166 - accuracy: 0.9957 - val_loss: 0.0329 - val_accuracy: 0.9940
Epoch 12/30
37800/37800 [==============================] - 172s 5ms/step - loss: 0.0170 - accuracy: 0.9952 - val_loss: 0.0325 - val_accuracy: 0.9940
Epoch 13/30
37800/37800 [==============================] - 568s 15ms/step - loss: 0.0169 - accuracy: 0.9953 - val_loss: 0.0319 - val_accuracy: 0.9943
Epoch 14/30
37800/37800 [==============================] - 180s 5ms/step - loss: 0.0158 - accuracy: 0.9958 - val_loss: 0.0321 - val_accuracy: 0.9938
Epoch 15/30
37800/37800 [==============================] - 179s 5ms/step - loss: 0.0156 - accuracy: 0.9959 - val_loss: 0.0321 - val_accuracy: 0.9938
Epoch 16/30
37800/37800 [==============================] - 179s 5ms/step - loss: 0.0148 - accuracy: 0.9963 - val_loss: 0.0324 - val_accuracy: 0.9936
Epoch 17/30
37800/37800 [==============================] - 181s 5ms/step - loss: 0.0163 - accuracy: 0.9953 - val_loss: 0.0328 - val_accuracy: 0.9938
Epoch 18/30
37800/37800 [==============================] - 180s 5ms/step - loss: 0.0153 - accuracy: 0.9958 - val_loss: 0.0328 - val_accuracy: 0.9936
Epoch 19/30
37800/37800 [==============================] - 184s 5ms/step - loss: 0.0159 - accuracy: 0.9958 - val_loss: 0.0331 - val_accuracy: 0.9938
Epoch 20/30
37800/37800 [==============================] - 181s 5ms/step - loss: 0.0137 - accuracy: 0.9963 - val_loss: 0.0329 - val_accuracy: 0.9938
Epoch 21/30
37800/37800 [==============================] - 180s 5ms/step - loss: 0.0153 - accuracy: 0.9959 - val_loss: 0.0325 - val_accuracy: 0.9936
Epoch 22/30
37800/37800 [==============================] - 180s 5ms/step - loss: 0.0154 - accuracy: 0.9958 - val_loss: 0.0325 - val_accuracy: 0.9933
Epoch 23/30
37800/37800 [==============================] - 182s 5ms/step - loss: 0.0151 - accuracy: 0.9957 - val_loss: 0.0324 - val_accuracy: 0.9936
Epoch 24/30
37800/37800 [==============================] - 182s 5ms/step - loss: 0.0152 - accuracy: 0.9958 - val_loss: 0.0328 - val_accuracy: 0.9938
Epoch 25/30
37800/37800 [==============================] - 183s 5ms/step - loss: 0.0134 - accuracy: 0.9966 - val_loss: 0.0323 - val_accuracy: 0.9940
Epoch 26/30
37800/37800 [==============================] - 182s 5ms/step - loss: 0.0145 - accuracy: 0.9959 - val_loss: 0.0323 - val_accuracy: 0.9943
Epoch 27/30
37800/37800 [==============================] - 187s 5ms/step - loss: 0.0130 - accuracy: 0.9965 - val_loss: 0.0326 - val_accuracy: 0.9938
Epoch 28/30
37800/37800 [==============================] - 182s 5ms/step - loss: 0.0137 - accuracy: 0.9960 - val_loss: 0.0324 - val_accuracy: 0.9940
Epoch 29/30
37800/37800 [==============================] - 183s 5ms/step - loss: 0.0139 - accuracy: 0.9961 - val_loss: 0.0329 - val_accuracy: 0.9936
Epoch 30/30
37800/37800 [==============================] - 184s 5ms/step - loss: 0.0151 - accuracy: 0.9958 - val_loss: 0.0326 - val_accuracy: 0.9933
<keras.callbacks.callbacks.History at 0x7f8bf7b7c350>
y_pred = conv_model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
my_submission = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})
my_submission.to_csv('dig_submission.csv', index=False)
sriram2397. “Digit-Recognizer-Kaggle/digit_recognizer.Ipynb at Master · SRIRAM2397/Digit-Recognizer-Kaggle.” GitHub. Accessed February 6, 2022. https://github.com/sriram2397/digit-recognizer-kaggle/blob/master/Digit_Recognizer.ipynb. 
