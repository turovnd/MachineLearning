# Linear regression
[Table of contents](https://github.com/fedy95/MachineLearning/blob/master/README.md)

## Used software:
- Python-IDE: PyCharm Community 2017.2;
- Project Interpreter: python 3.5.3 (amd64);
- Used python packeges:
	- matplotlib v2.1.0;
	- tabulate v0.8.1;
	- numpy v1.13.3.

### Problem
1) Реализовать *линейную регрессию*;
2) Настроить вектор коэффициентов двумя способами - *градиентным спуском* и *генетическим алгоритмом*;
3) Для оценки качества работы использовать *среднеквадратичное отклонение/ошибку* MSE;
4) Выборать гиперпараметры и методы произвольно;
5) Perform data visualization;
6) Требуется научить свой код принимать откуда-нибудь (лучше с консоли) дополнительные входные точки для проверки уже обученной модели.

Генетика:
![генетика](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/7.%20evolution.jpg)
При реализации эволюционного алгоритма особью является вектор коэффициентов Θ. Для хорошего результата достаточно делать один вид мутации - добавление к вектору коэффициентов случайного, нормально распределенного шума (в Python сгенерировать случайный вектор из нормального распределения можно с помощью функции `numpy.random.randn`[fn:4]). Можно даже без скрещивания. С размером потомства и процентом выживаемости можно поэкспериментировать, экспериментально хорошо работает увеличение популяции в 6 раз и выживаемость 1/6 популяции.
Если вы чувствуете в себе силы, в качестве эволюционного алгоритма можно выбрать алгоритм дифференциальной эволюции[fn:5]. Он сходится лучше, чем наивная эволюция.
Как работает ваш эволюционный алгоритм? Как вы подбирали параметры для него?
Ответ: Как написали, так и отвечайте. Задача может решаться всевозможными эволюционными алгоритмами, описание одного из вариантов реализации можно увидеть в подпункте Hints. Гиперпараметры (размер потомства, процент выживаемости и пр.) можно попытаться подобрать с помощью кросс-валидации, но на деле лучше всего работает метод “от фонаря”.

- начальная популяция: рандомная генерация [w0],[w1],[w2],[w3],[w4],[w5],[w6] [w(i)] = w0,w1,w2
- особь: [w(i)]
- хромосома:  0,1234567890123456 = float[-1; +1]
- ген: число в хромосоме
- отбор
	**(!) минимальная MSE**
	- размер потомства 
		
		![размер потомства](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/5.%20evolution.jpg)
	- процент выживаемости: 3 лучших особи
- формирование нового поколения:
	- скрещивание (кроссовер): однородный кроссовер: у ребенка первые 3 гена от лучшей особи, после чередуются с другим родителем через одну.
		
		![однородный кроссовер](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/6.%20evolution.jpg)
	
	- мутация: 
		- вероятность 5% изменения особи рандомной хромосомы (w0 [0-33] w1 (33-66) w2 [66-100]) рандомных 6 генов 
		- лучшая особь предыдущего шага не мутирует с вероятностью 100% (элитарность)
		- если особи совпадают на первых 0,1234567 генов, то лучшая остается, а остальные мутируют с вероятностью 100%: рандомное изменение на 13 генов (катаклизм)
 - критерий остановки: 13 катаклизмов пройдено

### Start dataset
[Dataset.txt](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/dataset.txt) - dependence of objects: area, number of rooms, price.

### Program structure
- [DatasetProcessing](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/DatasetProcessing.py):
	- getDataset;
	- getSeparetedData;
	- getAvgData;
	- getStandardDeviationData;
	- getNormalizeDataset;
	- getNormalizeInputDataset;
	- getCombinedInputData.
- [GradientDescent](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/GradientDescent.py):
	- calculateGradientDescent;
	- calculateInputPrice.
- [Visualization](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/Visualization.py):
	- build3DStartDataset;
	- build3DRegressionLinearPlusInput;
	- build3DRegressionLinear;
	- build3DCostFunction;
	- build2DInfo;
	- build2DTopMSEEvolution;
	- build2DIndividualMSEEvolution.
- [Evolution](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/Evolution.py):
	- startEvolution;
	- testEvo;
	- weightStartIntialization;
	- calculateMSE;
	- selection;
	- crossing.
- *presentation files*:
	- [main](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/main.py).

### Output
https://habrastorage.org/files/9e7/ec4/164/9e7ec41641d74a9dbcb696eeb60c1ec2.png

[other results](https://docs.google.com/spreadsheets/d/1_fdJo6_bG0gLd3Ci8oq-1gmV49EXWts24C2ImHvbD2g/edit#gid=303528850)

[excel table](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/results.xlsx)

Gradient:

![3DStartDataset](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/1.%203DStartDataset.jpg)
![3DCostFunction](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/2.%203DCostFunction.jpg)
![3DRegressionLinear](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/3.%203DRegressionLinear.jpg)
![2DInfo](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/4.%202DInfo.jpg)

Evolution:
![build2DTopMSEEvolution](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/8.%20build2DTopMSEEvolution.jpg)

![build2DIndividualMSEEvolution](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/images/9.%20build2DIndividualMSEEvolution.jpg)

### FAQ

![\left\{\begin{matrix} \widehat{y_i{}}=w_0{}x_0{}+w_1{}x_1{}+w_2{}x_2{}+\cdots+w_N{}x_N{} \\ x_0{}=1=const\\ \end{matrix}\right.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Cwidehat%7By_i%7B%7D%7D%3Dw_0%7B%7Dx_0%7B%7D&plus;w_1%7B%7Dx_1%7B%7D&plus;w_2%7B%7Dx_2%7B%7D&plus;%5Ccdots&plus;w_N%7B%7Dx_N%7B%7D%20%5C%5C%20x_0%7B%7D%3D1%3Dconst%5C%5C%20%5Cend%7Bmatrix%7D%5Cright.)

![\left\{\begin{matrix} x=\begin{Bmatrix}x_1{},x_2{},\cdots ,x_N{} \end{Bmatrix}^T\\
y=\begin{Bmatrix}y_1{},y_2{},\cdots ,y_N{} \end{Bmatrix}^T\\ w=\begin{Bmatrix}w_0{},w_1{},\cdots ,w_D{} \end{Bmatrix}^T\\ \end{matrix}\right.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%3D%5Cbegin%7BBmatrix%7Dx_1%7B%7D%2Cx_2%7B%7D%2C%5Ccdots%20%2Cx_N%7B%7D%20%5Cend%7BBmatrix%7D%5ET%5C%5C%20y%3D%5Cbegin%7BBmatrix%7Dy_1%7B%7D%2Cy_2%7B%7D%2C%5Ccdots%20%2Cy_N%7B%7D%20%5Cend%7BBmatrix%7D%5ET%5C%5C%20w%3D%5Cbegin%7BBmatrix%7Dw_0%7B%7D%2Cw_1%7B%7D%2C%5Ccdots%20%2Cw_D%7B%7D%20%5Cend%7BBmatrix%7D%5ET%5C%5C%20%5Cend%7Bmatrix%7D%5Cright.)

![\left\{\begin{matrix} w_i{}=w_i{}-\alpha\frac{\partial}{\partial w_i{}}J(w)\\ \frac{\partial}{\partial w_i{}}J(w)=\frac{\partial}{\partial w_i{}}\frac{1} {2*N}\sum_{i=1}^{N}(\widehat{y_i{}}-y_i{})^2=\frac{[x^T]{}([x][w]-[y])}{N}\\ \end{matrix}\right.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20w_i%7B%7D%3Dw_i%7B%7D-%5Calpha%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w_i%7B%7D%7DJ%28w%29%5C%5C%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w_i%7B%7D%7DJ%28w%29%3D%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w_i%7B%7D%7D%5Cfrac%7B1%7D%20%7B2*N%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28%5Cwidehat%7By_i%7B%7D%7D-y_i%7B%7D%29%5E2%3D%5Cfrac%7B%5Bx%5ET%5D%7B%7D%28%5Bx%5D%5Bw%5D-%5By%5D%29%7D%7BN%7D%5C%5C%20%5Cend%7Bmatrix%7D%5Cright.)

![\left\{\begin{matrix} [w]=[w]-\alpha[x^T{}]([\widehat{y}]&-&[y])\\ [\widehat{y}]=[x][w]\\ \end{matrix}\right.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Bw%5D%3D%5Bw%5D-%5Calpha%5Bx%5ET%7B%7D%5D%28%5B%5Cwidehat%7By%7D%5D%26-%26%5By%5D%29%5C%5C%20%5B%5Cwidehat%7By%7D%5D%3D%5Bx%5D%5Bw%5D%5C%5C%20%5Cend%7Bmatrix%7D%5Cright.)

![[w]=[w]-\alpha[X^T{}]([x][w]-&[y])](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Bw%5D%3D%5Bw%5D-%5Calpha%5BX%5ET%7B%7D%5D%28%5Bx%5D%5Bw%5D-%26%5By%5D%29)

![\begin{pmatrix} w_1{}\\ w_2{}\\ \vdots\\ w_D{}\\ \end{pmatrix}=\begin{pmatrix} w_1{}\\ w_2{}\\ \vdots\\
w_D{}\\ \end{pmatrix}-\alpha \begin{pmatrix} 1& 1& \cdots & 1&\\ x_{11}&  x_{21}&  \cdots & x_{N1}\\ x_{12}&  x_{22}&  \cdots & x_{N2}\\ \vdots &  \vdots &  \ddots & \vdots \\ x_{1D}&  x_{2D}&  \cdots & x_{ND}
\end{pmatrix}\begin{pmatrix} \begin{pmatrix} 1& x_{11}&  x_{12}&  \cdots & x_{1D}\\ 1& x_{21}&  x_{22}&  \cdots & x_{2D}\\ \vdots &  \vdots & \vdots & \ddots & \vdots \\ 1& x_{N1}&  x_{N2}&  \cdots & x_{ND} \end{pmatrix}& \begin{pmatrix} w_1{}\\ w_2{}\\ \vdots\\ w_D{}\\ \end{pmatrix} -&  \begin{pmatrix} y_1{}\\ y_2{}\\ \vdots\\ y_N{}\\ \end{pmatrix} \end{pmatrix}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Cbegin%7Bpmatrix%7D%20w_1%7B%7D%5C%5C%20w_2%7B%7D%5C%5C%20%5Cvdots%5C%5C%20w_D%7B%7D%5C%5C%20%5Cend%7Bpmatrix%7D%3D%5Cbegin%7Bpmatrix%7D%20w_1%7B%7D%5C%5C%20w_2%7B%7D%5C%5C%20%5Cvdots%5C%5C%20w_D%7B%7D%5C%5C%20%5Cend%7Bpmatrix%7D-%5Calpha%20%5Cbegin%7Bpmatrix%7D%201%26%201%26%20%5Ccdots%20%26%201%26%5C%5C%20x_%7B11%7D%26%20x_%7B21%7D%26%20%5Ccdots%20%26%20x_%7BN1%7D%5C%5C%20x_%7B12%7D%26%20x_%7B22%7D%26%20%5Ccdots%20%26%20x_%7BN2%7D%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20x_%7B1D%7D%26%20x_%7B2D%7D%26%20%5Ccdots%20%26%20x_%7BND%7D%20%5Cend%7Bpmatrix%7D%5Cbegin%7Bpmatrix%7D%20%5Cbegin%7Bpmatrix%7D%201%26%20x_%7B11%7D%26%20x_%7B12%7D%26%20%5Ccdots%20%26%20x_%7B1D%7D%5C%5C%201%26%20x_%7B21%7D%26%20x_%7B22%7D%26%20%5Ccdots%20%26%20x_%7B2D%7D%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%201%26%20x_%7BN1%7D%26%20x_%7BN2%7D%26%20%5Ccdots%20%26%20x_%7BND%7D%20%5Cend%7Bpmatrix%7D%26%20%5Cbegin%7Bpmatrix%7D%20w_1%7B%7D%5C%5C%20w_2%7B%7D%5C%5C%20%5Cvdots%5C%5C%20w_D%7B%7D%5C%5C%20%5Cend%7Bpmatrix%7D%20-%26%20%5Cbegin%7Bpmatrix%7D%20y_1%7B%7D%5C%5C%20y_2%7B%7D%5C%5C%20%5Cvdots%5C%5C%20y_N%7B%7D%5C%5C%20%5Cend%7Bpmatrix%7D%20%5Cend%7Bpmatrix%7D)
1) **Question:**
   Что такое *линейная регрессия*?
   
   **Answer:**
   Линейная регрессия — метод восстановления зависимости между двумя переменными.
   
2) **Question:**
   Что такое *среднеквадратичное отклонение/ошибка*?
   
   **Answer:**
   
   ![Q(a,X)=\frac{1}{l}\sum_{i=1}^{l}(a(x_{i})-y_{i})^{2}.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20Q%28a%2CX%29%3D%5Cfrac%7B1%7D%7Bl%7D%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%28a%28x_%7Bi%7D%29-y_%7Bi%7D%29%5E%7B2%7D.)
   
   Это *функционал ошибки*. Его преимущество (по сравнению с модулем отклюнения алгоритма/прогноза) заключается в том, что *квадрат отклонения алгоритма от истинного ответа* ![(a(x)-y)^{2}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%28a%28x%29-y%29%5E%7B2%7D), содержащийся в нем, является гладкой функцией (имеет производную во всех точках), что позволит использовать минимизацию градиентными методами.
   
   Для линейной модели (вместо ![(a(x_{i})](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20a%28x_%7Bi%7D%29), подставляя ![(\left \langle w,x_{i} \right \rangle](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%5Cleft%20%5Clangle%20w%2Cx_%7Bi%7D%20%5Cright%20%5Crangle) получается не функционал, а *функция*:
   
   ![Q(w,X)=\frac{1}{l}\sum_{i=1}^{l}(\left \langle w,x_{i} \right \rangle-y_{i})^{2}.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20Q%28w%2CX%29%3D%5Cfrac%7B1%7D%7Bl%7D%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%28%5Cleft%20%5Clangle%20w%2Cx_%7Bi%7D%20%5Cright%20%5Crangle-y_%7Bi%7D%29%5E%7B2%7D.)
   
   Ошибка зависит не от некоторой функции ![(a(x_{i})](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20a%28x_%7Bi%7D%29), а от вектора весов ![w](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20w), который возможно подвергнуть оптимизации.
   
3) **Question:**
   Как записать в матричном виде среднеквадратичную ошибку?
   
   **Answer:**   
   
   Среднеквадратичная ошибка в матричном виде:
   
   ![Q(w,X)=\frac{1}{l}\left \| Xw-y \right \|^{2}\rightarrow \min_{w}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20Q%28w%2CX%29%3D%5Cfrac%7B1%7D%7Bl%7D%5Cleft%20%5C%7C%20Xw-y%20%5Cright%20%5C%7C%5E%7B2%7D%5Crightarrow%20%5Cmin_%7Bw%7D), где:
   
   ![X=\begin{pmatrix}
 {x_{11}}& ...& {x_{1d}}\\ 
 ...&  ...& ...\\ 
 {x_{l1}}&  ...& {x_{ld}}
\end{pmatrix}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20X%3D%5Cbegin%7Bpmatrix%7D%20%7Bx_%7B11%7D%7D%26%20...%26%20%7Bx_%7B1d%7D%7D%5C%5C%20...%26%20...%26%20...%5C%5C%20%7Bx_%7Bl1%7D%7D%26%20...%26%20%7Bx_%7Bld%7D%7D%20%5Cend%7Bpmatrix%7D) - матрица "объекты-признаки".
   
   ![\begin{pmatrix}
{x_{11}} &  ...& {x_{1d}} 
\end{pmatrix}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%5Cbegin%7Bpmatrix%7D%20%7Bx_%7B11%7D%7D%20%26%20...%26%20%7Bx_%7B1d%7D%7D%20%5Cend%7Bpmatrix%7D) - все признаки i-ого объекта.
   
   ![\begin{pmatrix}
{x_{11}}\\ 
...\\ 
{x_{l1}}
\end{pmatrix}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%5Cbegin%7Bpmatrix%7D%20%7Bx_%7B11%7D%7D%5C%5C%20...%5C%5C%20%7Bx_%7Bl1%7D%7D%20%5Cend%7Bpmatrix%7D) - значения j-ого признака на всех объектах.

   ![y=\begin{bmatrix}
{y_{1}}\\ 
...\\ 
{y_{l}}
\end{bmatrix}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20y%3D%5Cbegin%7Bbmatrix%7D%20%7By_%7B1%7D%7D%5C%5C%20...%5C%5C%20%7By_%7Bl%7D%7D%20%5Cend%7Bbmatrix%7D) -    вектор ответов.
   
4) **Question:**
   Зачем использовать градиентный спуск, если существует способ получения аналитического решения задачи минимизации среднеквадратичной ошибки без оптимизации?
   
   Аналитическое решение: ![{w_{*}}=(X^{T}X)^{-1}X^{T}y.](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%7Bw_%7B*%7D%7D%3D%28X%5E%7BT%7DX%29%5E%7B-1%7DX%5E%7BT%7Dy.)
   
   **Answer:**   
   Основная проблема состоит в том, что необходимо обращать матрицу ![d\times d](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20d%5Ctimes%20d) (число признаков на признаков) - число операций ![{d^{3}}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%7Bd%5E%7B3%7D%7D). На тысяче признаков - критическая сложность вычислений.
   
   Кроме того, при обращении могут возникнуть численные проблемы при определенном устройстве матриц.
   
5) **Question:**
   Что такое градиентый спуск?
   	
   **Answer:**
   Градиентый спуск - это способ поиска решения оптимизационным подходом.
   
   Подход основан на том, что среднеквадратичная ошибка обладает свойствами выпуклой и гладкой функции.
   
   Из выпуклости функции следует, что у функции существует один минимум, а из гладкости, что в каждой точки функции возможно вычислить градиент.
   
   Градиент - это вектор, указывающий направление наибольшего возрастания некоторой величины.
   
   Другими словами, **градиент** показывает сторону наискорейшего возрастания функции, а **антиградиент** показывает сторону наискорейшего убавания функции.

   Алгоритм:
   
   1. Инициализация: 
   
   	Каким-то образом находится начальное приближение для вектора весов: либо вектор заполняется нулями ![w](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20w): ![w^0=0](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20w%5E0%3D0), либо случайными небольшими числами.

   2. Цикл t: 
   
	   ![w^{t}=w^{t-1}-{\eta_{t}}\bigtriangledown Q(w^{t-1},X)](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20w%5E%7Bt%7D%3Dw%5E%7Bt-1%7D-%7B%5Ceta_%7Bt%7D%7D%5Cbigtriangledown%20Q%28w%5E%7Bt-1%7D%2CX%29), где

	   ![\bigtriangledown Q(w^{t-1},X)](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%5Cbigtriangledown%20Q%28w%5E%7Bt-1%7D%2CX%29) - вектор градиентов этой точки;

	   ![{\eta_{t}}](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%7B%5Ceta_%7Bt%7D%7D) - шаг, регулирует, на сколько далеко шагаем в сторону антиградиента (оптимальный шаг не слишком большой).

	   Градиентные шаги повторяются до наступления сходимости.

	   Сходимость можно определять:
	   - как ситуацию, когда векторы весов от шага к шагу меняются не очень сильно: 
			![\left \| w^{t}-w^{t-1} \right \|<\varepsilon ](http://latex.codecogs.com/svg.latex?%5Cfn_jvn%20%5Csmall%20%5Cleft%20%5C%7C%20w%5E%7Bt%7D-w%5E%7Bt-1%7D%20%5Cright%20%5C%7C%3C%5Cvarepsilon);
			
	   - производить сравнение функционала ошибки между текущей иттерацией и предыдущей.

2) **Question:**
   Настройка коэффециентов линейной регрессии генетикой
   
   **Answer:**
   
3) **Question:**
   Среднеквадратичная линейная регрессия
   
   **Answer:**

### Links
1) [линейная регрессия](http://www.machinelearning.ru/wiki/index.php?title=%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F_(%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80));
https://ru.coursera.org/learn/supervised-learning/lecture/hCGR6/obuchieniie-linieinoi-rieghriessii
https://basegroup.ru/deductor/function/algorithm/linear-regression
http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf
2) [London Machine Learning Stydy Group. Lecture 1](https://www.youtube.com/watch?v=v-LJxJlBxfc);
3) [Trace in linear algebra](https://en.wikipedia.org/wiki/Trace_(linear_algebra));
4) [Evolution algohithms [RUS]](https://www.youtube.com/watch?v=65AwXU7Z68c).
