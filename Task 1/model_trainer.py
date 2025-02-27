# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

from data_processor import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('model_trainer')

class ModelTrainer:
    """
    Handles model training, evaluation, and serialization
    for the RTV Risk Assessment Model.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelTrainer with configuration settings.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.data_processor = DataProcessor(config)
        self.model = None
        self.optimal_threshold = 0.5
        self.train_date = datetime.now().strftime("%Y-%m-%d")
        self.model_version = f"rtv_risk_model_v{datetime.now().strftime('%Y%m%d')}"
        self.output_dir = self.config.get('output_dir', 'models')

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self, data_file: str) -> None:
        """
        Train the model using a Random Forest classifier.

        Args:
            data_file: Path to the data file
        """
        logger.info(f"Starting model training with data from {data_file}")

        # Process data
        X_scaled, y = self.data_processor.process_data(data_file, for_training=True)
        
        # Ensure all data is numeric before training
        # Convert any remaining non-numeric columns to numeric
        if isinstance(X_scaled, pd.DataFrame):
            for col in X_scaled.columns:
                if X_scaled[col].dtype == 'object':
                    logger.warning(f"Converting non-numeric column '{col}' to numeric")
                    try:
                        # Try to convert to numeric, set errors to 'coerce' to replace non-convertible values with NaN
                        X_scaled[col] = pd.to_numeric(X_scaled[col], errors='coerce')
                        # Fill NaN values with column mean or 0
                        X_scaled[col] = X_scaled[col].fillna(X_scaled[col].mean() if not X_scaled[col].isna().all() else 0)
                    except Exception as e:
                        logger.error(f"Error converting column '{col}': {str(e)}")
                        # Drop problematic columns if they can't be converted
                        logger.warning(f"Dropping column '{col}' due to conversion issues")
                        X_scaled = X_scaled.drop(columns=[col])
        
        # Convert to numpy array if it's still a DataFrame
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values

        if y is None:
            raise ValueError("Target variable not found in the data")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )

        logger.info(f"Data split complete. Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train a Random Forest classifier with fixed parameters
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        logger.info("Model training complete")

        # Find optimal threshold
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate F1 score for different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []

        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_threshold, average='weighted')
            f1_scores.append(f1)

        # Find the threshold that maximizes F1 score
        self.optimal_threshold = thresholds[np.argmax(f1_scores)]

        logger.info(f"Optimal threshold found: {self.optimal_threshold:.2f}")

        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test

        # Store the actual feature names used for training
        if isinstance(X_scaled, pd.DataFrame):
            self.training_features = X_scaled.columns.tolist()
        else:
            self.training_features = self.data_processor.all_features

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Get predictions
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

        # Ensure arrays have the same length
        if len(self.y_test) != len(y_pred):
            logger.warning(f"Length mismatch: y_test ({len(self.y_test)}) != y_pred ({len(y_pred)})")
            # Adjust arrays to have the same length (use the shorter length)
            min_length = min(len(self.y_test), len(y_pred))
            self.y_test = self.y_test[:min_length]
            y_pred = y_pred[:min_length]
            y_pred_proba = y_pred_proba[:min_length]

        # Calculate metrics
        classification_rep = classification_report(self.y_test, y_pred, output_dict=True)
        conf_mat = confusion_matrix(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.training_features[:len(self.model.feature_importances_)],
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Store evaluation results
        evaluation_results = {
            'classification_report': classification_rep,
            'confusion_matrix': conf_mat,
            'roc_auc': roc_auc,
            'feature_importance': feature_importance,
            'optimal_threshold': self.optimal_threshold
        }

        # Log evaluation results
        logger.info(f"Model evaluation complete. ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Classification report:\n{classification_report(self.y_test, y_pred)}")

        # Generate plots
        self._generate_evaluation_plots(y_pred, y_pred_proba, feature_importance)

        return evaluation_results

    def _generate_evaluation_plots(self, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                  feature_importance: pd.DataFrame) -> None:
        """
        Generate and save evaluation plots.

        Args:
            y_pred: Predicted target values
            y_pred_proba: Predicted probabilities
            feature_importance: DataFrame with feature importance
        """
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        conf_mat = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['At Risk', 'Not At Risk'],
                   yticklabels=['At Risk', 'Not At Risk'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.model_version}_confusion_matrix.png"))
        plt.close()

        # ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(plots_dir, f"{self.model_version}_roc_curve.png"))
        plt.close()

        # Feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.model_version}_feature_importance.png"))
        plt.close()

        logger.info(f"Evaluation plots saved to {plots_dir}")

    def save_model(self) -> str:
        """
        Save the trained model and associated components.

        Returns:
            Path to the saved model file
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create model package with all necessary components
        model_package = {
            'model': self.model,
            'scaler': self.data_processor.scaler,
            'feature_names': self.training_features,
            'threshold': self.optimal_threshold,
            'train_date': self.train_date,
            'model_version': self.model_version
        }

        # Save the model package
        model_path = os.path.join(self.output_dir, f"{self.model_version}.joblib")
        joblib.dump(model_package, model_path)

        logger.info(f"Model saved to {model_path}")

        # Generate model documentation
        self._generate_model_card()

        return model_path

    def _generate_model_card(self) -> None:
        """
        Generate a model card with model details and evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Evaluate the model if not already done
        evaluation_results = self.evaluate()

        # Create model card
        model_card = f"""
            # Model Card: RTV Risk Assessment Model

            ## Model Overview
            - **Model Version**: {self.model_version}
            - **Training Date**: {self.train_date}
            - **Model Type**: Random Forest Classifier
            - **Purpose**: Predict households at risk of not achieving $2/day income target

            ## Model Parameters
            - n_estimators: 200
            - random_state: 42

            ## Model Performance
            - **ROC-AUC Score**: {evaluation_results['roc_auc']:.4f}
            - **Optimal Threshold**: {self.optimal_threshold:.2f}

            ### Classification Report
            ## Top 10 Important Features
            {evaluation_results['feature_importance'].head(10).to_string()}

            ## Model Limitations
            - The model has been trained on data from specific communities and may not generalize well to significantly different regions.
            - Seasonality effects are not fully captured by the current feature set.
            - The model should be monitored for drift as economic conditions change.

            ## Ethical Considerations
            - Predictions should be used as a guide for interventions, not as the sole decision criteria.
            - False positives (classifying a household as not-at-risk when they are) may lead to withholding needed assistance.
            - False negatives (classifying a household as at-risk when they are not) may lead to inefficient resource allocation.

            ## Recommendations
            - Re-train the model every 6 months with fresh data.
            - Collect additional features related to market access and economic opportunities.
            - Consider regional variations when interpreting model outputs.
          """

        # Save model card
        model_card_path = os.path.join(self.output_dir, f"{self.model_version}_card.md")
        with open(model_card_path, 'w') as f:
            f.write(model_card)

        logger.info(f"Model card saved to {model_card_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train RTV Risk Assessment Model')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save model files')

    args = parser.parse_args()

    # Configure trainer
    config = {
        'output_dir': args.output_dir
    }

    trainer = ModelTrainer(config)

    try:
        # Train model
        trainer.train(args.data_file)

        # Evaluate model
        evaluation_results = trainer.evaluate()

        # Save model
        model_path = trainer.save_model()

        logger.info(f"Training complete. Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")