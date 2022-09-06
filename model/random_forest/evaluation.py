
explainerModel = shap.TreeExplainer(forest)
samplesToExplain = X_test.sample(n=1000)
shapValues = explainerModel.shap_values(samplesToExplain)
averageShapValues = np.abs(shapValues[1]).mean(axis=0)

SHAPfeatureImportances['importances'] = averageShapValues
SHAPfeatureImportances.to_csv('shap_feature_importances.csv')
shap.summary_plot(shapValues[1], samplesToExplain)