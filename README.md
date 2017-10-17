Discipline "Intellectual systems and technologies"
---------------------------------------------------
Used software:
- Python-IDE: PyCharm Community 2017.2;
- Project Interpreter: python 3.5.3 (amd64);
- Used python packeges:
	- matplotlib v2.1.0;
	- tabulate v0.8.1.
--------------------------------------------------- 
1. KNN metric classifier.
  Dataset.txt - set of objects: coordinates of the dot(x,y),class{0,1}.
  ---
Program structure:
- Main: started point.
- OutputMethods:
	- outputTable.
- test:
	make_computing.
- Plot:
	- buildPlotWithAllDots;
	- buildPlotCentroid;
	- buildPlotCircle.
- Statistic:
	- compareClasses;
	- computingRecall;
	- computingSpecificity;
	- computingPrecision;
	- computingAccuracy;
	- computingF1_measure.
- DatasetProcessing:
	- getDataset;
	- getDotsByClass;
	- computingManhattanDistance2D;
	- computingManhattanDistance3D;
	- computingEuclideanDistance2D;
	- computingEuclideanDistance3D;
	- getCentroid;
	- classifyDotCentroid;
	- classifyDotCircle;
	- classifyKNNCentroid;
	- classifyKNNCircle;
	- getTrainTestDots.
 ---
Output table
 ---
| Group number | Training dots | Test dots | k (neighbors) | Kernel functions | Metrics for configuring kNN | Spatial coordinate transformations | F1-measure | Recall | Specificity | Precision | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | | none | manhattan | none | | | | | |
| | | | | gaussian | manhattan | none | | | | | |
| | | | | logistic | manhattan | none | | | | | |
| | | | | none | euclidean | none | | | | | |
| | | | | gaussian | euclidean | none | | | | | |
| | | | | logistic | euclidean | none | | | | | |
|-|-|-|-|-|-|-|-|-|-|-|-|
| | | | | none | manhattan | elliptic | | | | | |
| | | | | gaussian | manhattan | elliptic | | | | | |
| | | | | logistic | manhattan | elliptic | | | | | |
| | | | | none | euclidean | elliptic | | | | | |
| | | | | gaussian | euclidean | elliptic | | | | | |
| | | | | logistic | euclidean | elliptic | | | | | |
|-|-|-|-|-|-|-|-|-|-|-|-|
| | | | | none | manhattan | hyperbolic | | | | | |
| | | | | gaussian | manhattan | hyperbolic | | | | | |
| | | | | logistic | manhattan | hyperbolic | | | | | |
| | | | | none | euclidean | hyperbolic | | | | | |
| | | | | gaussian | euclidean | hyperbolic | | | | | |
| | | | | logistic | euclidean | hyperbolic | | | | | |

 ---
FAQ:
 ---
1) **Question:**
   What is [linear classifier](https://en.wikipedia.org/wiki/Linear_classifier)?
   
   **Answer:**
   It is a classification algorithm, which based on the construcion of a linear separating surface. In the case of two classes of separating force is a hyperplane that divides the feature space into two half-spaces. Metric classifier is based on the concept of similarity between objects. In this task, the similarity measure between objects is *distance*.

2) **Question:**
   What is [empirical risk](https://en.wikipedia.org/wiki/Empirical_risk_minimization) ([эмпирический риск](http://www.machinelearning.ru/wiki/index.php?title=%D0%AD%D0%BC%D0%BF%D0%B8%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D1%80%D0%B8%D1%81%D0%BA))?)
   
   **Answer:**
   This is the average error value of the algorithm on the training sample.

3) **Question:**
   How to determine the optimal number of neighbors (k_neighbors)?
   
   **Answer:**
   As a rule, lt's processed  how: sqrt(datasetStartSample). In this problem datasetStartSample=118 => optimal k_neighbors≈10. You can check this experimentally.

4) **Question:**
   What cross-validation is used?
   
   **Answer:**
   Was implemented [k-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation).

5) **Question:**
   How to determine the optimal number of k-fold (k_fold)?
   
   **Answer:**
   Experimentally. I was get the best f1-measure (0,905819) when k_fold was equal 9 (of course, because of the random *shuffle* function, this number may vary slightly) . Think that 10 is the optimal k_fold value.

6) **Question:**
   What does your decision contain??

   **Answer:** 
- 2 [Spatial coordinate transformations](https://en.wikipedia.org/wiki/Paraboloid):
	- [elliptic paraboloid](https://en.wikipedia.org/wiki/Paraboloid#Elliptic_paraboloid);
	- [hyperbolic paraboloid](https://en.wikipedia.org/wiki/Paraboloid#Hyperbolic_paraboloid).
- 2 [Kernel functions](https://en.wikipedia.org/wiki/Kernel_(statistics)):
	- [gaussian](https://en.wikipedia.org/wiki/Normal_distribution);
	- [logistic](https://en.wikipedia.org/wiki/Logistic_distribution).
- 2 Metrics for configuring kNN:
	- [manhattan distance (p=1)](https://en.wikipedia.org/wiki/Taxicab_geometry);
	- [euclidean distance (p=2)](https://en.wikipedia.org/wiki/Euclidean_distance).
- Quality assessment:
	- [Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Sensitivity) or [Recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall);
	- [Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Specificity);
	- [Precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision);
	- [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision);
	- [F1-measure](https://en.wikipedia.org/wiki/F1_score).

Tasks:
- Число фолдов для кросс-валидации определите и обоснуйте сами исходя из числа объектов в датасете.
- Можно попробовать несколько способов выбора k.
	- Рандомный выбор для тестирования
- Хотелось бы увидеть некоторую визуализацию данных.
	- Разделение и раскраска границ точек
- дерево
писать какие-нибудь формулы и рисовать примеры при обосновании выборов.

[RESULTS](https://docs.google.com/spreadsheets/d/1IkHaIzaHMTVHIrxbvIXl9kbQINCencgx8dtkAhNKXRw/edit#gid=0)

[Задача классификации](https://ru.wikipedia.org/wiki/%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0_%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8)
[Кластерный анализ](https://ru.wikipedia.org/wiki/%D0%9A%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7)
[Info](https://github.com/flyingleafe/ML-Course-ITMO/blob/master/Homework.org):

