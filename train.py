from lazypredict.Supervised import LazyClassifier
def lazy(X_train, X_test, y_train, y_test):
                import sklearn
                import lazypredict
                CLASSIFIERS=lazypredict.Supervised.CLASSIFIERS
                clf = LazyClassifier(verbose=0,predictions=True, custom_metric=None,classifiers=CLASSIFIERS[:10]+CLASSIFIERS[14:])
                models,predictions = clf.fit(X_train, X_test, y_train, y_test)
                print(models)
                return models,predictions
def random_forrest(X_train, X_test, y_train, y_test):
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import RandomizedSearchCV
                param_dist = {
                    'n_estimators': range(100, 1000, 100),  # Number of trees in the forest (adjust range as needed)
                    'max_depth': range(2, 20, 2),  # Maximum depth of each tree (adjust range as needed)
                    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features considered at each split
                    'min_samples_split': range(2, 11),  # Minimum samples required to split a node
                    'min_samples_leaf': range(1, 6)  # Minimum samples required at each leaf node
                }

                # Create a Random Forest classifier instance
                rfc = RandomForestClassifier(random_state=42)  # Set a random state for reproducibility

                # Create a RandomizedSearchCV object
                random_search = RandomizedSearchCV(estimator=rfc,
                                                   param_distributions=param_dist,
                                                   n_iter=10,  # Number of random parameter sets to try (adjust as needed)
                                                   cv=3)  # Number of cross-validation folds (adjust as needed)

                # Fit the random search to the training data
                random_search.fit(X_train, y_train)

                # Access the best hyperparameters found
                best_params = random_search.best_params_

                # Print the best hyperparameters
                print(f"Best Hyperparameters: {best_params}")

                # Create a new Random Forest model with the best hyperparameters
                best_model = RandomForestClassifier(**best_params)

                # Train the best model on the training data
                best_model.fit(X_train,  y_train)
                print("random_forrest:",best_model.score(X_test,y_test))
def svc(X_train, X_test, y_train, y_test):
            from sklearn.svm import SVC
            from sklearn.model_selection import GridSearchCV

            # Define the hyperparameter search space
            param_grid = {
                'C': [0.1, 1, 10],  # Regularization parameter
                'kernel': ['linear', 'rbf', 'poly'],  # Kernel function
                'gamma': [0.01, 0.1, 1],  # Kernel coefficient (for rbf and poly)
                'degree': [2, 3]  # Degree of polynomial kernel (if using 'poly')
            }

            # Create an SVC object
            svc = SVC(random_state=42)  # Set a random seed for reproducibility (optional)

            # Create a GridSearchCV object
            grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5)  # Adjust cv for cross-validation folds

            # Fit the GridSearchCV to the training data
            grid_search.fit(X_train, y_train)

            # Access the best hyperparameters
            best_params = grid_search.best_params_
            print(f"Best Hyperparameters: {best_params}")

            # Create a new SVC model with the best hyperparameters
            best_model = SVC(**best_params)

            # Train the best model on the training data
            best_model.fit(X_train, y_train)

            # Use the best_model for prediction on new data
            print("SVC",best_model.score(X_test,y_test))
def logistc(X_train, X_test, y_train, y_test):
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import RandomizedSearchCV

            # Define the hyperparameter search space
            param_dist = {
                'penalty': ['l1', 'l2'],  # Type of regularization
                'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
                'solver': ['liblinear', 'lbfgs'],  # Optimization algorithm
                'max_iter': [100, 200, 500]  # Maximum number of iterations
            }

            # Create a Logistic Regression object
            logistic_reg = LogisticRegression(random_state=42)  # Set a random seed for reproducibility (optional)

            # Create a RandomizedSearchCV object
            random_search = RandomizedSearchCV(estimator=logistic_reg,
                                               param_distributions=param_dist,
                                               n_iter=3,  # Adjust n_iter for more/less random evaluations
                                               cv=2)  # Adjust cv for more/less cross-validation folds

            # Fit the RandomizedSearchCV to the training data
            random_search.fit(X_train, y_train)

            # Access the best hyperparameters
            best_params = random_search.best_params_
            print(f"Best Hyperparameters: {best_params}")

            # Create a new Logistic Regression model with the best hyperparameters
            best_model = LogisticRegression(penalty=best_params['penalty'], C=best_params['C'], solver=best_params['solver'], max_iter=best_params['max_iter'])

            # Train the best model on the training data
            best_model.fit(X_train, y_train)

            # Use the best_model for prediction on new data
            print("LogisticRegression",best_model.score(X_test,y_test))