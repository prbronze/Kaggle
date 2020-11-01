from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_curve(estimator, X, yt):
    """Plot learning curves for each label of a multilabel problem.
    
    Parameters:
    -----------
    estimator : sklearn-type classifier
        Estimator used to as classifier.
   
    X : DataFrame/Matrix
        Features used to fit the estimator.
    
    yt : DataFrame
        DataFrame of multiple labels.
    """
    
    for col in yt.columns:
        y = yt[col]
        _, axes = plt.subplots(1, 1, figsize=(5, 5))
        
        axes.set_title(f'{col}')

        axes.set_xlabel("Training examples")
        axes.set_ylabel("Score")

        train_sizes, train_scores, test_scores = \
                    learning_curve(estimator,X,y,
                                   train_sizes=np.linspace(.1, 1.0, 5),
                                   cv=5,
                                   n_jobs=4,
                                   scoring='roc_auc') # uses stratified as default

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        axes.grid()
        axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes.legend(loc="best")
        plt.show()