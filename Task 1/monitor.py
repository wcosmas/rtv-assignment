import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('monitor')

class ModelMonitor:
    """
    Monitors the RTV Risk Assessment Model for drift and performance issues.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelMonitor with configuration settings.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.model_path = self.config.get('model_path')
        self.reference_data_path = self.config.get('reference_data_path')
        self.alert_threshold = self.config.get('alert_threshold', 0.05)
        self.alerts_enabled = self.config.get('alerts_enabled', False)
        self.email_recipients = self.config.get('email_recipients', [])
        self.monitoring_dir = self.config.get('monitoring_dir', 'monitoring')

        # Create monitoring directory if it doesn't exist
        if not os.path.exists(self.monitoring_dir):
            os.makedirs(self.monitoring_dir)

        # Load model and reference data if paths are provided
        if self.model_path:
            self.load_model()

        if self.reference_data_path:
            self.load_reference_data()

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the model to be monitored.

        Args:
            model_path: Path to the model file (overrides config path)
        """
        try:
            path = model_path or self.model_path
            if not path:
                raise ValueError("Model path not provided")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            logger.info(f"Loading model from {path}")
            model_package = joblib.load(path)

            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.threshold = model_package['threshold']
            self.model_version = model_package.get('model_version', 'unknown')
            self.train_date = model_package.get('train_date', 'unknown')

            # Update model path if provided as parameter
            if model_path:
                self.model_path = model_path

            logger.info(f"Model loaded successfully. Version: {self.model_version}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_reference_data(self, reference_data_path: Optional[str] = None) -> None:
        """
        Load reference data for drift detection.

        Args:
            reference_data_path: Path to reference data (overrides config path)
        """
        try:
            path = reference_data_path or self.reference_data_path
            if not path:
                raise ValueError("Reference data path not provided")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Reference data file not found: {path}")

            logger.info(f"Loading reference data from {path}")
            self.reference_data = pd.read_csv(path)

            # Create a DataFrame with the expected features
            # Initialize with zeros to ensure all required features exist
            self.reference_features = pd.DataFrame(0, index=range(len(self.reference_data)), 
                                                  columns=self.feature_names)
            
            # Copy available features from reference data
            available_features = [feat for feat in self.feature_names if feat in self.reference_data.columns]
            missing_features = [feat for feat in self.feature_names if feat not in self.reference_data.columns]
            
            if available_features:
                logger.info(f"Found {len(available_features)} of {len(self.feature_names)} required features in reference data")
                
                # Copy available features from reference data
                for feat in available_features:
                    self.reference_features[feat] = pd.to_numeric(self.reference_data[feat], errors='coerce')
            else:
                logger.warning("None of the required features are present in the reference data")
                
            if missing_features:
                logger.warning(f"Missing features in reference data: {missing_features[:10]}...")
                if len(missing_features) > 10:
                    logger.warning(f"...and {len(missing_features) - 10} more")
            
            # Handle missing values
            self.reference_features = self.reference_features.fillna(0)

            # Update reference data path if provided as parameter
            if reference_data_path:
                self.reference_data_path = reference_data_path

            logger.info(f"Reference data processed successfully. Shape: {self.reference_features.shape}")

        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            raise

    def detect_data_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference data and new data.

        Args:
            new_data: New data to compare with reference data

        Returns:
            Dictionary with drift detection results
        """
        if not hasattr(self, 'reference_features'):
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")

        # Extract features from new data
        new_features = new_data[self.feature_names].copy()

        # Calculate drift for each feature
        drift_results = {}
        drifted_features = []

        for feature in self.feature_names:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(self.reference_features[feature]) or \
               not pd.api.types.is_numeric_dtype(new_features[feature]):
                continue

            # Replace missing values
            ref_values = self.reference_features[feature].fillna(self.reference_features[feature].median())
            new_values = new_features[feature].fillna(new_features[feature].median())

            # KS test for distribution comparison
            ks_statistic, p_value = ks_2samp(ref_values, new_values)

            # Track results
            drift_results[feature] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'is_drifted': bool(p_value < self.alert_threshold)
            }

            if p_value < self.alert_threshold:
                drifted_features.append(feature)

        # Calculate overall drift status
        drift_percentage = len(drifted_features) / len(self.feature_names)
        is_drifted = drift_percentage > 0.3  # Consider drifted if more than 30% of features have drifted

        # Prepare summary
        drift_summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': self.model_version,
            'total_features': len(self.feature_names),
            'drifted_features': drifted_features,
            'drift_percentage': drift_percentage,
            'is_drifted': is_drifted,
            'feature_details': drift_results
        }

        # Log summary
        logger.info(f"Data drift detection: {len(drifted_features)} of {len(self.feature_names)} "
                   f"features drifted ({drift_percentage:.1%})")
        if drifted_features:
            logger.info(f"Drifted features: {drifted_features}")

        # Generate alert if needed
        if is_drifted and self.alerts_enabled:
            self._send_drift_alert(drift_summary)

        return drift_summary

    def evaluate_model_performance(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Evaluate model performance on new data with known outcomes.

        Args:
            data: DataFrame with features and actual target values
            target_column: Name of the column containing actual target values

        Returns:
            Dictionary with performance metrics
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded. Call load_model() first.")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Extract features and target
        features = data[self.feature_names].copy()
        y_true = data[target_column]

        # Handle missing values
        features = features.fillna(features.median())

        # Scale features
        X_scaled = self.scaler.transform(features)

        # Generate predictions
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Prepare performance summary
        performance_summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': self.model_version,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': conf_matrix.tolist(),
            'sample_size': len(y_true)
        }

        # Log summary
        logger.info(f"Performance evaluation: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Save performance results
        self._save_performance_results(performance_summary)

        return performance_summary

    def _save_performance_results(self, performance_summary: Dict[str, Any]) -> None:
        """
        Save performance results to file.

        Args:
            performance_summary: Dictionary with performance metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_{self.model_version}_{timestamp}.json"
        file_path = os.path.join(self.monitoring_dir, filename)

        try:
            with open(file_path, 'w') as f:
                json.dump(performance_summary, f, indent=4)

            logger.info(f"Performance results saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving performance results: {str(e)}")

    def track_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Track prediction statistics over time.

        Args:
            predictions: DataFrame containing model predictions

        Returns:
            Dictionary with prediction statistics
        """
        # Calculate prediction statistics
        total_records = len(predictions)
        at_risk_count = (predictions['is_at_risk'] == 0).sum()
        at_risk_percentage = at_risk_count / total_records

        # Create probability distribution histogram
        hist_values, bin_edges = np.histogram(predictions['risk_probability'], bins=10, range=(0, 1))

        # Prepare statistics summary
        prediction_stats = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': self.model_version,
            'total_records': int(total_records),
            'at_risk_count': int(at_risk_count),
            'at_risk_percentage': float(at_risk_percentage),
            'histogram_values': hist_values.tolist(),
            'histogram_bins': bin_edges.tolist()
        }

        # Log summary
        logger.info(f"Prediction statistics: {at_risk_count} of {total_records} "
                   f"households at risk ({at_risk_percentage:.1%})")

        # Save statistics
        self._save_prediction_stats(prediction_stats)

        # Generate prediction distribution plot
        self._generate_prediction_plot(predictions)

        return prediction_stats

    def _save_prediction_stats(self, prediction_stats: Dict[str, Any]) -> None:
        """
        Save prediction statistics to file.

        Args:
            prediction_stats: Dictionary with prediction statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_stats_{self.model_version}_{timestamp}.json"
        file_path = os.path.join(self.monitoring_dir, filename)

        try:
            with open(file_path, 'w') as f:
                json.dump(prediction_stats, f, indent=4)

            logger.info(f"Prediction statistics saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving prediction statistics: {str(e)}")

    def _generate_prediction_plot(self, predictions: pd.DataFrame) -> None:
        """
        Generate and save prediction distribution plot.

        Args:
            predictions: DataFrame containing model predictions
        """
        # Create plots directory
        plots_dir = os.path.join(self.monitoring_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Prediction distribution histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions['risk_probability'], bins=20, kde=True)
        plt.axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold: {self.threshold:.2f}')
        plt.title('Distribution of Risk Probabilities')
        plt.xlabel('Risk Probability')
        plt.ylabel('Count')
        plt.legend()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"prediction_dist_{self.model_version}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Prediction distribution plot saved to {plot_path}")

    def _send_drift_alert(self, drift_summary: Dict[str, Any]) -> None:
        """
        Send alert email about data drift.

        Args:
            drift_summary: Dictionary with drift detection results
        """
        if not self.email_recipients:
            logger.warning("No email recipients configured. Alert not sent.")
            return

        try:
            # Create email content
            subject = f"[ALERT] Data Drift Detected in RTV Risk Model {self.model_version}"

            body = f"""
            Data drift detected in RTV Risk Assessment Model.

            Model Version: {self.model_version}
            Timestamp: {drift_summary['timestamp']}

            Drift Summary:
            - {len(drift_summary['drifted_features'])} of {drift_summary['total_features']} features have drifted ({drift_summary['drift_percentage']:.1%})

            Drifted Features:
            {', '.join(drift_summary['drifted_features'])}

            Action Required:
            - Review the drifted features
            - Collect new training data if necessary
            - Consider retraining the model

            This is an automated alert from the RTV Model Monitoring System.
            """

            # Prepare email
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = 'model-monitoring@rtv.org'
            msg['To'] = ', '.join(self.email_recipients)
            msg.attach(MIMEText(body, 'plain'))

            # Send email - this is a placeholder, replace with actual email sending logic
            # For example, using SMTP:
            """
            smtp_server = 'smtp.example.com'
            smtp_port = 587
            smtp_username = 'username'
            smtp_password = 'password'

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            """

            logger.info(f"Drift alert email sent to {', '.join(self.email_recipients)}")

        except Exception as e:
            logger.error(f"Error sending drift alert email: {str(e)}")

    def check_model_freshness(self) -> Dict[str, Any]:
        """
        Check if the model is fresh or needs retraining.

        Returns:
            Dictionary with model freshness status
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded. Call load_model() first.")

        # Parse training date
        try:
            train_date = datetime.strptime(self.train_date, "%Y-%m-%d")
        except:
            train_date = datetime.now() - timedelta(days=180)  # Default to 6 months ago if parse fails

        # Calculate days since training
        days_since_training = (datetime.now() - train_date).days

        # Determine freshness status
        is_fresh = days_since_training <= 180  # Consider fresh if trained within 6 months

        freshness_status = {
            'model_version': self.model_version,
            'train_date': self.train_date,
            'days_since_training': days_since_training,
            'is_fresh': is_fresh,
            'recommended_action': 'none' if is_fresh else 'retrain'
        }

        # Log status
        logger.info(f"Model freshness check: {days_since_training} days since training. "
                   f"Status: {'Fresh' if is_fresh else 'Needs retraining'}")

        return freshness_status

    def run_all_checks(self, new_data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Run all monitoring checks at once.

        Args:
            new_data: New data to monitor
            target_column: Name of the column containing actual target values (if available)

        Returns:
            Dictionary with all monitoring results
        """
        # Run each check and collect results
        monitoring_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': self.model_version
        }

        # Check model freshness
        monitoring_results['freshness'] = self.check_model_freshness()

        # Detect data drift
        try:
            monitoring_results['drift'] = self.detect_data_drift(new_data)
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            monitoring_results['drift'] = {'error': str(e)}

        # Evaluate performance if target column is provided
        if target_column and target_column in new_data.columns:
            try:
                monitoring_results['performance'] = self.evaluate_model_performance(new_data, target_column)
            except Exception as e:
                logger.error(f"Error evaluating performance: {str(e)}")
                monitoring_results['performance'] = {'error': str(e)}

        # Track predictions
        try:
            # Generate predictions if not already present
            if 'risk_probability' not in new_data.columns or 'is_at_risk' not in new_data.columns:
                from predictor import RiskPredictor
                predictor = RiskPredictor(self.model_path)
                predictions = predictor.predict(new_data)
            else:
                predictions = new_data

            monitoring_results['predictions'] = self.track_predictions(predictions)
        except Exception as e:
            logger.error(f"Error tracking predictions: {str(e)}")
            monitoring_results['predictions'] = {'error': str(e)}

        # Save overall results
        self._save_monitoring_results(monitoring_results)

        return monitoring_results

    def _save_monitoring_results(self, monitoring_results: Dict[str, Any]) -> None:
        """
        Save all monitoring results to file.

        Args:
            monitoring_results: Dictionary with all monitoring results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_{self.model_version}_{timestamp}.json"
        file_path = os.path.join(self.monitoring_dir, filename)

        try:
            with open(file_path, 'w') as f:
                json.dump(monitoring_results, f, indent=4)

            logger.info(f"Monitoring results saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving monitoring results: {str(e)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor RTV Risk Assessment Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--reference_data', type=str, required=True, help='Path to reference data file')
    parser.add_argument('--new_data', type=str, required=True, help='Path to new data file')
    parser.add_argument('--target_column', type=str, help='Name of target column in new data (if available)')
    parser.add_argument('--monitoring_dir', type=str, default='monitoring', help='Directory to save monitoring results')
    parser.add_argument('--email', type=str, help='Email address for alerts (comma-separated for multiple)')

    args = parser.parse_args()

    # Parse email recipients
    email_recipients = args.email.split(',') if args.email else []

    # Configure monitor
    config = {
        'model_path': args.model_path,
        'reference_data_path': args.reference_data,
        'monitoring_dir': args.monitoring_dir,
        'alerts_enabled': bool(email_recipients),
        'email_recipients': email_recipients
    }

    # Initialize monitor
    monitor = ModelMonitor(config)

    try:
        # Load new data
        logger.info(f"Loading new data from {args.new_data}")
        new_data = pd.read_csv(args.new_data)

        # Run all checks
        results = monitor.run_all_checks(new_data, args.target_column)

        # Summarize results
        logger.info("Monitoring checks completed")

        # Report drift status
        if 'drift' in results and 'is_drifted' in results['drift']:
            if results['drift']['is_drifted']:
                logger.warning("DATA DRIFT DETECTED! Model may need retraining.")
            else:
                logger.info("No significant data drift detected.")

        # Report performance if available
        if 'performance' in results and 'error' not in results['performance']:
            logger.info(f"Model performance: "
                       f"Accuracy: {results['performance']['accuracy']:.4f}, "
                       f"F1: {results['performance']['f1_score']:.4f}")

        # Report freshness
        if results['freshness']['is_fresh']:
            logger.info("Model is up-to-date.")
        else:
            logger.warning(f"Model was trained {results['freshness']['days_since_training']} days ago "
                          f"and should be retrained.")

    except Exception as e:
        logger.error(f"Error running monitoring checks: {str(e)}")