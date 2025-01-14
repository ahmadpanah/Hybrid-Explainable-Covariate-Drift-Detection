import shap
import lime
import lime.lime_tabular

class ExplanationService:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def explain_with_shap(self):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.data)
        shap.summary_plot(shap_values, self.data)

    def explain_with_lime(self, instance):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.data.values, 
            feature_names=self.data.columns, 
            class_names=["No Drift", "Drift"], 
            mode="regression"
        )
        exp = explainer.explain_instance(instance, self.model.predict, num_features=5)
        exp.show_in_notebook()