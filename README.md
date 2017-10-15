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
- OutputMethods;
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
	- getManhattanDistance;
	- getEuclideanDistance;
	- getCentroid;
	- classifyDotCentroid;
	- classifyDotCircle;
	- classifyKNNCentroid;
	- classifyKNNCircle.
 ---
Output table
 ---
| Started dot groups | k (neighbors) | Kernel functions | Metrics for configuring kNN | Spatial coordinate transformations | F1-measure | Recall | Specificity | Precision | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | none | manhattan | none | | | | | |
| | | gaussian | manhattan | none | | | | | |
| | | logistic | manhattan | none | | | | | |
| | | none | euclidean | none | | | | | |
| | | gaussian | euclidean | none | | | | | |
| | | logistic | euclidean | none | | | | | |
|-|-|-|-|-|-|-|-|-|-|
| | | none | manhattan | elliptic | | | | | |
| | | gaussian | manhattan | elliptic | | | | | |
| | | logistic | manhattan | elliptic | | | | | |
| | | none | euclidean | elliptic | | | | | |
| | | gaussian | euclidean | elliptic | | | | | |
| | | logistic | euclidean | elliptic | | | | | |
|-|-|-|-|-|-|-|-|-|-|
| | | none | manhattan | hyperbolic | | | | | |
| | | gaussian | manhattan | hyperbolic | | | | | |
| | | logistic | manhattan | hyperbolic | | | | | |
| | | none | euclidean | hyperbolic | | | | | |
| | | gaussian | euclidean | hyperbolic | | | | | |
| | | logistic | euclidean | hyperbolic | | | | | |

- 2 Spatial coordinate transformations:
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



Число фолдов для кросс-валидации определите и обоснуйте сами исходя из числа объектов в датасете.
Можно попробовать несколько способов выбора k.
- Рандомный выбор для тестирования
Хотелось бы увидеть некоторую визуализацию данных.
- Разделение и раскраска границ точек


писать какие-нибудь формулы и рисовать примеры при обосновании выборов.

- (+-)разбиение по файлам
- дерево

