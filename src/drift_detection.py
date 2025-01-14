from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import IsolationForest

class DriftDetectionService:
    def __init__(self, reference_data, current_data):
        self.reference_data = reference_data
        self.current_data = current_data

    def detect_statistical_drift(self):
        drift_results = {}
        for feature in self.reference_data.columns:
            if self.reference_data[feature].dtype == "float64":
                ks_stat, ks_pvalue = ks_2samp(self.reference_data[feature], self.current_data[feature])
                drift_results[feature] = {"KS Statistic": ks_stat, "KS p-value": ks_pvalue}
            elif self.reference_data[feature].dtype == "object":
                contingency_table = pd.crosstab(self.reference_data[feature], self.current_data[feature])
                chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency_table)
                drift_results[feature] = {"Chi-Square Statistic": chi2_stat, "Chi-Square p-value": chi2_pvalue}
        return drift_results

    def detect_ml_drift(self):
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(self.reference_data)
        predictions = model.predict(self.current_data)
        return np.mean(predictions == -1)  # Percentage of anomalies

    def hybrid_drift_detection(self, alpha=0.5):
        statistical_results = self.detect_statistical_drift()
        statistical_drift_score = np.mean([result["KS Statistic"] for result in statistical_results.values()])
        ml_drift_score = self.detect_ml_drift()
        hybrid_score = alpha * statistical_drift_score + (1 - alpha) * ml_drift_score
        return hybrid_score, statistical_results, ml_drift_score