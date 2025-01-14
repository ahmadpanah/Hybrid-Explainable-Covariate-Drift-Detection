import json
import numpy as np
import pandas as pd
from src.data_ingestion import DataIngestionLayer
from src.drift_detection import DriftDetectionService
from src.explanation_service import ExplanationService
from src.notification_layer import NotificationAndActionLayer

def lambda_handler(event, context):
    try:
        # Parse the event to get the dataset name
        dataset_name = event.get("dataset", "CreditCard")  # Default to CreditCard dataset

        # Step 1: Data Ingestion
        data_ingestion = DataIngestionLayer(dataset_name)
        reference_data, current_data = train_test_split(data_ingestion.preprocessed_data, test_size=0.3, random_state=42)

        # Step 2: Drift Detection
        drift_detection = DriftDetectionService(reference_data, current_data)
        hybrid_score, statistical_results, ml_drift_score = drift_detection.hybrid_drift_detection()

        # Step 3: Explanation
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(reference_data)
        explanation_service = ExplanationService(model, current_data)
        explanation_service.explain_with_shap()
        explanation_service.explain_with_lime(current_data.iloc[0])

        # Step 4: Notification and Action
        notification_layer = NotificationAndActionLayer(hybrid_score, 0.9)  # Confidence score set to 0.9
        alert_level = notification_layer.generate_alert()
        notification_layer.trigger_actions(alert_level)

        # Return results
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Drift detection completed successfully.",
                "hybrid_score": hybrid_score,
                "statistical_results": statistical_results,
                "ml_drift_score": ml_drift_score,
                "alert_level": alert_level
            })
        }
    except Exception as e:
        # Handle errors
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "An error occurred during drift detection.",
                "error": str(e)
            })
        }

# For local testing (optional)
if __name__ == "__main__":
    # Simulate an AWS Lambda event
    event = {
        "dataset": "CreditCard"  # Change to "PaySim" or "IoT-23" for other datasets
    }
    context = {}  # Context is not used in this example
    result = lambda_handler(event, context)
    print(result)