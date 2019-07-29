# ANN-MLP-Scikit-learn

Multi-Layer Perceptron (MLP) as Regressor:

<p align="center">
  <img width=450 src="mlp-network.png"/>
 </p>


Neural network parameters:

- Number of hidden layers and number of neurons per layers
- Penalty (Alpha)
- Initial learning rate
- Learning rate ('constant', 'invscaling', 'adaptive')
- Solver ('lbfgs', 'sgd', 'adam')
- Moment of the descending gradient (Momentun) -- if we use the 'sgd' solver

  'lbfgs' is an optimizer in the family of quasi-Newton methods.
  'sgd' refers to stochastic gradient descent.
  'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

Note: there are much more parameters, these are considered the most important

More information: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html


Optimization methods tested to search the hyperparameters space are:

* Exhaustive Grid Search
* Randomized Parameter Optimization

More information: https://scikit-learn.org/stable/modules/grid_search.html

Dependences:

    python - Scikit-learn
    python - Pandas
    python - NumPy
    python - Matplolib
    python - Statsmodels





