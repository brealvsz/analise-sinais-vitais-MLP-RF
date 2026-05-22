from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from config import SEED


def treinar_regressao(X_train, y_train):
    param_grid = {
        'n_estimators':    [50, 100, 200],
        'max_depth':       [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print(f"[Regressão] Melhores hiperparâmetros: {gs.best_params_}")
    return gs.best_estimator_


def treinar_classificacao(X_train, y_train):
    param_grid = {
        'n_estimators':  [50, 100, 200],
        'max_depth':     [None, 10, 20],
        'criterion':     ['gini', 'entropy'],
        'min_samples_leaf': [1, 2],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print(f"[Classificação] Melhores hiperparâmetros: {gs.best_params_}")
    return gs.best_estimator_
