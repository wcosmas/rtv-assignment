import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
from typing import Dict, Union, List, Optional, Any
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('predictor')

class RiskPredictor:
    """
    Handles prediction using the trained RTV Risk Assessment Model.
    """

    def __init__(self, model_path: str):
        """
        Initialize the RiskPredictor with a trained model.

        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.load_model()

    def load_model(self) -> None:
        """
        Load the trained model and associated components.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading model from {self.model_path}")
            model_package = joblib.load(self.model_path)

            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.threshold = model_package['threshold']
            self.model_version = model_package.get('model_version', 'unknown')
            self.train_date = model_package.get('train_date', 'unknown')

            logger.info(f"Model loaded successfully. Version: {self.model_version}, Training date: {self.train_date}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data for prediction.

        Args:
            df: Input DataFrame

        Returns:
            Scaled feature matrix as numpy array
        """
        try:
            # Get the exact number of features expected by the scaler
            expected_feature_count = self.scaler.n_features_in_
            logger.info(f"Scaler expects {expected_feature_count} features")
            
            if len(self.feature_names) != expected_feature_count:
                logger.warning(f"Feature name count ({len(self.feature_names)}) doesn't match scaler's expected feature count ({expected_feature_count})")
                # Adjust feature_names to match the expected count if needed
                if len(self.feature_names) > expected_feature_count:
                    logger.info(f"Trimming feature list to match scaler's expectations")
                    self.feature_names = self.feature_names[:expected_feature_count]
                else:
                    # This case is unlikely but handled for completeness
                    logger.warning(f"Not enough feature names provided. Using available names and padding with generic names.")
                    missing_count = expected_feature_count - len(self.feature_names)
                    self.feature_names.extend([f'feature_{i}' for i in range(missing_count)])
            
            # Create a new DataFrame with only the required features
            # Initialize with zeros to ensure all required features exist
            X = pd.DataFrame(0, index=range(len(df)), columns=self.feature_names)
            
            # Log which features from the model are available in the input data
            available_features = [feat for feat in self.feature_names if feat in df.columns]
            missing_features = [feat for feat in self.feature_names if feat not in df.columns]
            
            if not available_features:
                logger.warning("None of the required features are present in the input data")
                logger.warning("Using default values (0) for all features")
            else:
                logger.info(f"Found {len(available_features)} of {len(self.feature_names)} required features")
                
                # Copy available features from input data
                for feat in available_features:
                    X[feat] = pd.to_numeric(df[feat], errors='coerce')
            
            if missing_features:
                logger.warning(f"Missing features in input data: {missing_features[:10]}...")
                if len(missing_features) > 10:
                    logger.warning(f"...and {len(missing_features) - 10} more")
            
            # Handle missing values
            X = X.fillna(0)  # Use 0 as default for missing values
            
            # Convert to numpy array before scaling to avoid feature name issues
            X_array = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X_array)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            # Print more detailed information for debugging
            logger.error(f"DataFrame shape: {df.shape}, columns sample: {list(df.columns)[:5]}")
            logger.error(f"Expected features: {self.feature_names[:5]} (showing first 5 of {len(self.feature_names)})")
            logger.error(f"Scaler expects {getattr(self.scaler, 'n_features_in_', 'unknown')} features")
            raise

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate risk predictions for the input data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with original data plus risk predictions and probabilities
        """
        try:
            # Preprocess data
            X_scaled = self.preprocess_data(df)

            # Generate predictions
            risk_probabilities = self.model.predict_proba(X_scaled)[:, 1]
            risk_predictions = (risk_probabilities >= self.threshold).astype(int)

            # Add predictions to original data
            df_result = df.copy()
            df_result['risk_probability'] = risk_probabilities
            df_result['is_at_risk'] = risk_predictions

            # Add risk category for easier interpretation
            df_result['risk_category'] = pd.cut(
                df_result['risk_probability'],
                bins=[0, 0.3, 0.7, 1],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )

            # Add prediction metadata
            df_result['prediction_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_result['model_version'] = self.model_version

            # Log prediction summary
            at_risk_count = (risk_predictions == 0).sum()
            total_count = len(risk_predictions)
            logger.info(f"Predictions generated for {total_count} households. "
                       f"Households at risk: {at_risk_count} ({at_risk_count/total_count:.1%})")

            return df_result

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def predict_single(self, household_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk prediction for a single household.

        Args:
            household_data: Dictionary containing household data

        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([household_data])

        # Generate prediction
        result_df = self.predict(df)

        # Convert result to dictionary
        result = {
            'is_at_risk': bool(result_df['is_at_risk'].iloc[0] == 0),  # True if at risk (0)
            'risk_probability': float(result_df['risk_probability'].iloc[0]),
            'risk_category': str(result_df['risk_category'].iloc[0]),
            'prediction_timestamp': result_df['prediction_timestamp'].iloc[0],
            'model_version': result_df['model_version'].iloc[0]
        }

        return result

    def save_predictions(self, df_predictions: pd.DataFrame, output_file: str) -> None:
        """
        Save prediction results to file.

        Args:
            df_predictions: DataFrame with predictions
            output_file: Path to save predictions
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save predictions
            df_predictions.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_version': self.model_version,
            'train_date': self.train_date,
            'threshold': self.threshold,
            'feature_count': len(self.feature_names),
            'top_features': sorted(zip(self.feature_names, self.model.feature_importances_),
                                 key=lambda x: x[1], reverse=True)[:10]
        }

def batch_predict(model_path: str, input_file: str, output_file: str) -> None:
    """
    Run batch prediction on a data file.

    Args:
        model_path: Path to the model file
        input_file: Path to the input data file
        output_file: Path to save prediction results
    """
    try:
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        # Initialize predictor
        predictor = RiskPredictor(model_path)

        # Generate predictions
        df_predictions = predictor.predict(df)

        # Save predictions
        predictor.save_predictions(df_predictions, output_file)

        # Log summary
        at_risk_count = (df_predictions['is_at_risk'] == 0).sum()
        total_count = len(df_predictions)
        logger.info(f"Batch prediction complete. Total: {total_count}, "
                   f"At risk: {at_risk_count} ({at_risk_count/total_count:.1%})")

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate predictions with RTV Risk Assessment Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output_file', type=str, help='Path to output predictions file')

    args = parser.parse_args()

    # Default output file if not specified
    if not args.output_file:
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")

    try:
        # Run batch prediction
        batch_predict(args.model_path, args.input_file, args.output_file)

    except Exception as e:
        logger.error(f"Error: {str(e)}")