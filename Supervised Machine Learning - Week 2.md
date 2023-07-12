## <mark class="hltr-red">Vectorization Videos</mark>

For vectorized product, you can use `np.dot(x,y)`

## <mark class="hltr-red">NumPy Vectorization Lab</mark>

NumPy Documentation including a basic introduction: [NumPy.org](https://numpy.org/doc/stable/)
A challenging feature topic: [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

### Vector Creation

For 1-D array, the shape is `(n,)`, and can be checked by `arr.shape`
For getting the data type of an array, you can use `arr.dtype`

#### # NumPy routines which allocate memory and fill arrays with value
>A vector with float values of `0.` can be created using `np.zeros()`
>A vector with random values between 0 and 1 is created by `np.random.random_sample()`

#### # NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument 
>A vector with continuous values at regular intervals can be created using `np.arange()`
>A vector with random values between 0 and 1 is created by `np.random.rand()`

#### # NumPy routines which allocate memory and fill with user specified values
>A vector with custom values can be created using `np.array()`

### Operations on Vectors

NumPy provides a very complete set of indexing and slicing capabilities: [Slicing and Indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html)

There are single vector operations like `-arr`, `np.sum()`, `np.mean()`, `arr**2` and `5 * arr`

The shapes of two vectors must be appropriate for vector vector element-wise operations

You can capture time with `time.time()`
The f-strings of python for float values up to a certain decimal point `{1000*(t):.4f}`

### Matrices

We can create matrices conveniently using `np.arange(8).reshape(-1, 4)`

You can access one specific element rather than a row using `arr[2, 0]` instead of conventional `arr[2][0]`

You can use slicing in this way: `arr[:, 2:7:1]`

## <mark class="hltr-red">Gradient Descent for Multiple Linear Regression</mark>

Instead of linear regression, some libraries use a complicated method, only useful for linear regression case to solve for `w` and `b` without iterations. This method is called the **normal equation**

However, mostly gradient descent is still the recommended method for finding the parameters

## <mark class="hltr-red">Feature Scaling and Learning Rate</mark>

In order to get the mean of each column you can use `np.mean(arr, axis = 0)`

In order to get the standard deviation of each column you can use `np.std(arr, axis = 0)`

In order to get the normalized values for each column, you can subtract the mean column wise and divide by the standard deviation column wise by using `(arr - mu) / sigma`

Normalization can speed up gradient descent a lot and we can take larger values of *alpha*, such as 0.1, which can be a good starting value

Before predicting new values, you must normalize the input data using the previously obtained mean and standard deviation values that you stored.

## <mark class="hltr-red">Feature Engineering and Polynomial Regression</mark>

It can be defined as using intuition or knowledge to design new features, by transforming or combining original features.

You can add engineered features efficiently using `np.c_[x, x**2, x**3]`
`np.c_` concatenates along the column axis and converts each into a row, effectively creating a 2d-array which has the new features.

We can get peak to peak range by column in a dataset by `np.ptp(arr, axis = 0)`

After performing feature engineering, always remember to perform feature scaling as the scales are quite different.

## <mark class="hltr-red">Linear Regression using Scikit-Learn</mark>

[sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) will perform z-score normalization, which is referred to as 'standard score'.
```
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```

Scikit-learn has a gradient descent regression model [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#examples-using-sklearn-linear-model-sgdregressor), which performs best with normalized inputs.
```
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
```
We can get the number of iterations it actually completed using `.n_iter_` and the number of weights updated using `.t_`
We can get the parameters associated with the *normalized* input data using `.coef_` for the weights and `.intercept_` for the bias.

We can obtain the predicted values using the `.predict(X_norm)` method.
