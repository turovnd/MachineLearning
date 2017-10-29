# Linear regression
[Table of contents](https://github.com/fedy95/MachineLearning/blob/master/README.md)

## Used software:
- Python-IDE: PyCharm Community 2017.2;
- Project Interpreter: python 3.5.3 (amd64);
- Used python packeges:
	- matplotlib v2.1.0;
	- tabulate v0.8.1.

### Problem
1) Требуется настроить коэффициенты линейной регрессии двумя способами: *градиентным спуском* и *генетическим алгоритмом*. Выбор гиперпараметров и конкретных методов настройки оставляю за вами, но будьте готовы ответить на доп вопросы по ним;
2) Perform data visualization;
3) Для оценки качества работы используем среднеквадратичное отклонение MSE;
4) Требуется научить свой код принимать откуда-нибудь (лучше с консоли) дополнительные входные точки для проверки уже обученной модели.

### Start dataset
[Dataset.txt](https://github.com/fedy95/MachineLearning/blob/master/2.%20Linear%20regression/dataset.txt) - dependence of objects: area, number of rooms, price.

### Program structure

### Output table

### FAQ
1) **Question:**
   Настройка коэффециентов линейной регрессии градиентым спуском
   	**градиент** показывает сторону наискорейшего возрастания функции
	**антиградиент** показывает сторону наискорейшего убавания функции
	
   	выпусклость функции => у функции один минимум
	градкость функции => в каждой точки функции возможно вычислть градиент
	
	инициализация: каким-то образом находится начальное приближение для вектора весов w: w^0=0
		заполнить вектор весов нулями или случайными небольшими числами.
	в цикле: на iой итерации берется приближение веса с предыдущей итерации, вычитается вектор градиента в этой точке, умноженный на коэффи...
	до сходимости
	
	Сходимость можно определять:
		векторы весов от шага к шагу меняются не очень сильно (< епсилон коэффициент) - завершение алгоритма
		сравнение функционала ошибки между текущей иттерацией и предыдущей
   
   матрица объектов-признаков, вектор ответов
   **Answer:**

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
