import numpy as np

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

class ModelScore():
    def __init__(self, identifier: str, y_test: np.array, y_pred: np.array):
        self.identifier = identifier
        self.acc = accuracy_score(y_test, y_pred)
        self.rec = recall_score(y_test, y_pred, average='macro')
        self.pre = precision_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')

    @property
    def results(self):
        return self.identifier, self.acc, self.rec, self.pre, self.f1 

class ModelEval:
    def __init__(
        self, 
        identifier: str,
        model, 
        param_grid: dict[list], 
        X_train: np.array, 
        y_train: np.array, 
        X_test: np.array, 
        y_test: np.array):

        self.identifier = identifier 
        self._model = model 
        self._param_grid = param_grid 
        self._X_train = X_train
        self._y_train = y_train 
        self._X_test = X_test 
        self._y_test = y_test
    
    @property 
    def results(self):
        return self.score.results 
        
    def evaluate(self):
        '''Perform grid search and return '''

        # Set up GridSearchCV 
        self.grid_search = GridSearchCV(
            estimator=self._model,
            param_grid=self._param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1)

        # Train the model
        self.grid_search.fit(self._X_train, self._y_train)
        self.best_clf = self.grid_search.best_estimator_

        # Pred
        y_pred = self.best_clf.predict(self._X_test)

        # Get stats
        self.score = ModelScore(self.identifier, self._y_test, y_pred)


def evaluate_model(
    identifier: str, 
    clf, 
    param_grid: dict, 
    X_train: np.array, 
    y_train: np.array, 
    X_test: np.array, 
    y_test: np.array):
    '''Evaluate a given model'''

    # Set up eval
    model_eval = ModelEval(identifier, clf, param_grid, X_train, y_train, X_test, y_test)
    model_eval.evaluate()

    return model_eval 




   

