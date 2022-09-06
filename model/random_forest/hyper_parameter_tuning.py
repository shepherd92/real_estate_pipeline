def initializeLogFile(filename: str) -> None:

    logFile = open(filename, 'w')
    logFile.write('Dataset,classWeight,nEstimators,maxDepth,minSamplesSplit,minSamplesLeaf,maxFeatures,bootstrap,CV,AUC\n')
    logFile.close()


def gridSearchCVScores(paramGrid: dict, logFilename: str) -> None:

    logFile = open(logFilename, 'a', buffering=1)
    CV = 5

    maxAUC = 0.5

    allIterations = len(paramGrid["classWeight"    ]) * \
                    len(paramGrid["nEstimators"    ]) * \
                    len(paramGrid["bootstrap"      ]) * \
                    len(paramGrid["maxFeatures"    ]) * \
                    len(paramGrid["maxDepth"       ]) * \
                    len(paramGrid["minSamplesSplit"]) * \
                    len(paramGrid["minSamplesLeaf" ])

    iterationsDone = 0

    for classWeight in paramGrid["classWeight"]:
        for nEstimators in paramGrid["nEstimators"]:
            for bootstrap in paramGrid["bootstrap"]:
                for maxFeatures in paramGrid["maxFeatures"]:
                    for maxDepth in paramGrid["maxDepth"]:
                        for minSamplesSplit in paramGrid["minSamplesSplit"]:
                            for minSamplesLeaf in paramGrid["minSamplesLeaf"]:

                                forest = RandomForestClassifier(class_weight      = classWeight,
                                                                n_estimators      = nEstimators,
                                                                bootstrap         = bootstrap,
                                                                max_features      = maxFeatures,
                                                                max_depth         = maxDepth,
                                                                min_samples_split = minSamplesSplit,
                                                                min_samples_leaf  = minSamplesLeaf)

                                cvScores = cross_val_score(forest, X_train, y_train, scoring='roc_auc', cv=CV, n_jobs=6, verbose=0)
                                AUC = mean(cvScores)

                                maxAUC = max(AUC, maxAUC)

                                logFile.write(f"{paramGrid['dataset']}," +
                                              f"{classWeight},"          +
                                              f"{nEstimators},"          +
                                              f"{maxDepth},"             +
                                              f"{minSamplesSplit},"      +
                                              f"{minSamplesLeaf},"       +
                                              f"{maxFeatures},"          +
                                              f"{bootstrap},"            +
                                              f"{CV},"                   +
                                              f"{AUC:.6}\n")

                                iterationsDone += 1

                                print(f"\r{100 * iterationsDone / allIterations:.2f}% ({iterationsDone} / {allIterations}); maximal AUC: {maxAUC:.4f}", flush=True, end='')

    logFile.close()


def randomSearchCVScores(paramGrid: dict, iterations: int, logFilename: str) -> None:
    logFile = open(logFilename, 'a', buffering=1)
    CV = 5

    maxAUC = 0.5

    for _ in range(iterations):
        classWeight     = random.choice(paramGrid["classWeight"    ])
        nEstimators     = random.choice(paramGrid["nEstimators"    ])
        bootstrap       = random.choice(paramGrid["bootstrap"      ])
        maxFeatures     = random.choice(paramGrid["maxFeatures"    ])
        maxDepth        = random.choice(paramGrid["maxDepth"       ])
        minSamplesSplit = random.choice(paramGrid["minSamplesSplit"])
        minSamplesLeaf  = random.choice(paramGrid["minSamplesLeaf" ])

        forest = RandomForestClassifier(class_weight      = classWeight,
                                        n_estimators      = nEstimators,
                                        bootstrap         = bootstrap,
                                        max_features      = maxFeatures,
                                        max_depth         = maxDepth,
                                        min_samples_split = minSamplesSplit,
                                        min_samples_leaf  = minSamplesLeaf)

        cvScores = cross_val_score(forest, X_train, y_train, scoring='roc_auc', cv=CV, n_jobs=6, verbose=0)
        AUC = mean(cvScores)

        maxAUC = max(AUC, maxAUC)

        logFile.write(f"{paramGrid['dataset']}," +
                      f"{classWeight         }," +
                      f"{nEstimators         }," +
                      f"{maxDepth            }," +
                      f"{minSamplesSplit     }," +
                      f"{minSamplesLeaf      }," +
                      f"{maxFeatures         }," +
                      f"{bootstrap           }," +
                      f"{CV                  }," +
                      f"{AUC:.6              }\n")

    logFile.close()


# parameter space
paramGrid = {
    "dataset"        : 'all 50 topics',
    "classWeight"    : ['balanced'],
    "nEstimators"    : [500, 1000],
    "maxFeatures"    : ['log2'],  # 'log2', 'sqrt',
    "maxDepth"       : [10, 20, 50, 100],
    "minSamplesSplit": [10, 20, 50, 100],
    "minSamplesLeaf" : [10, 20, 50, 100],
    "bootstrap"      : [True]
}

LOG_FILE = 'random_forest.log'
ALGORITHM = 'grid'  # 'random'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# initializeLogFile(LOG_FILE)
if   ALGORITHM == 'grid'  :
    gridSearchCVScores(paramGrid, LOG_FILE)
elif ALGORITHM == 'random':
    randomSearchCVScores(paramGrid, 1000, LOG_FILE)

algorithm = 'random'  # random, grid

param_grid = {
    "n_estimators"     : n_estimators,
    "max_features"     : max_features,
    "max_depth"        : max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf" : min_samples_leaf,
    "bootstrap"        : bootstrap
}

logFile = open('random_forest_january_data.log', 'a')
forest = RandomForestClassifier(class_weight='balanced')

# define scoring function 
def custom_auc(ground_truth, predictions):
    # I need only one column of predictions["0" and "1"]. You can get an error here
    # while trying to return both columns at once
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    fpr, tpr, _ = roc_curve(ground_truth, predictions, pos_label=1)
    AUC = auc(fpr, tpr)
    logFile.write(f'AUC={AUC}, ', end='', flush=True)
    return AUC

myAUC = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)

if algorithm == 'random':
    randomCV = RandomizedSearchCV(forest, param_grid, n_iter=1000, cv=5, n_jobs=1, verbose=2, scoring=myAUC)
    randomCV.fit(X_train, y_train)
    print(randomCV.best_params_)
elif algorithm == 'grid':
    gridCV = GridSearchCV        (forest, param_grid,              cv=5, n_jobs=1, verbose=2, scoring=myAUC)
    gridCV.fit(X_train, y_train)
    print(gridCV.best_params_)

logFile.close()