### To apply an classifier on this data, we need to flatten the image, to
### turn the data in a (samples, feature) matrix:
##n_samples = len(X_train)
##X = X_train.reshape((n_samples, -1))
##y = y_train

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_fraction=0.5, random_state=0)

### Set the parameters by cross-validation
##tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
##                     'C': [1, 10, 100, 1000]},
##                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Set the parameters by cross-validation
tuned_parameters = [{'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['optimal'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['constant'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l1'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['l2'], 'rho' : ['None'], 'warm_start' : ['True']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : [0.85], 'warm_start' : ['False']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : ['None'], 'warm_start' : ['False']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : [0.85], 'warm_start' : ['True']},
                    {'learning_rate': ['invscaling'], 'alpha': [1e-3, 1e-4], 'n_iter': [1, 5, 10, 20], 'penalty' : ['elasticnet'], 'rho' : ['None'], 'warm_start' : ['True']}]

scores = [('precision', precision_score), ('recall', recall_score)]

for score_name, score_func in scores:
    print "# Tuning hyper-parameters for %s" % score_name
    print

    clf = GridSearchCV(SGDClassifier(loss="log", shuffle = True, penalty = 'l2'), tuned_parameters, score_func=score_func)
##    clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=score_func)
    clf.fit(x_train, Y_train, cv=5)

    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)
    print

    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = Y_test, clf.predict(x_test)
    print classification_report(y_true, y_pred)
    print
