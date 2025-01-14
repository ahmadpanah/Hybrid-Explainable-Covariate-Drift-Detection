class NotificationAndActionLayer:
    def __init__(self, drift_score, confidence_score):
        self.drift_score = drift_score
        self.confidence_score = confidence_score

    def generate_alert(self):
        if self.drift_score > 0.8 and self.confidence_score >= 0.9:
            return "Critical"
        elif 0.5 < self.drift_score <= 0.8 and self.confidence_score >= 0.7:
            return "High"
        elif 0.3 < self.drift_score <= 0.5 and self.confidence_score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def trigger_actions(self, alert_level):
        if alert_level == "Critical":
            print("Triggering model retraining and stakeholder notification.")
        elif alert_level == "High":
            print("Triggering hyperparameter optimization and logging.")
        elif alert_level == "Medium":
            print("Triggering feature engineering review.")
        else:
            print("No action required.")