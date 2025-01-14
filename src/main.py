from sklearn.model_selection import train_test_split

class HybridExplainableDriftDetectionFramework:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_ingestion = DataIngestionLayer(dataset_name)
        self.reference_data, self.current_data = train_test_split(self.data_ingestion.preprocessed_data, test_size=0.3, random_state=42)
        self.drift_detection = DriftDetectionService(self.reference_data, self.current_data)
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(self.reference_data)

    def run(self):
        # Drift Detection
        hybrid_score, statistical_results, ml_drift_score = self.drift_detection.hybrid_drift_detection()
        print(f"Hybrid Drift Score: {hybrid_score}")
        print(f"Statistical Results: {statistical_results}")
        print(f"ML Drift Score: {ml_drift_score}")

        # Explanation
        explanation_service = ExplanationService(self.model, self.current_data)
        explanation_service.explain_with_shap()
        explanation_service.explain_with_lime(self.current_data.iloc[0])

        # Notification and Action
        notification_layer = NotificationAndActionLayer(hybrid_score, 0.9)  # Confidence score set to 0.9
        alert_level = notification_layer.generate_alert()
        print(f"Alert Level: {alert_level}")
        notification_layer.trigger_actions(alert_level)

# Run the framework
if __name__ == "__main__":
    framework = HybridExplainableDriftDetectionFramework("CreditCard")  # Change to "PaySim" or "IoT-23"
    framework.run()