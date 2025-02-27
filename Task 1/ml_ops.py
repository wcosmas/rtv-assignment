import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
import subprocess
import sys
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ml_ops')

class MLOpsPipeline:
    """
    Orchestrates the end-to-end MLOps pipeline for RTV Risk Assessment Model.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MLOps pipeline with configuration.

        Args:
            config_path: Path to configuration file (JSON)
        """
        # Default configuration
        self.config = {
            'data_file': 'interview_dataset.csv',
            'output_dir': 'models',
            'predictions_dir': 'predictions',
            'monitoring_dir': 'monitoring',
            'reference_data_ratio': 0.3,  # Portion of data to use as reference
            'notification_email': None,
            'retrain_frequency_days': 180,  # Retrain every 6 months
        }

        # Load config from file if provided
        if config_path:
            self._load_config(config_path)

        # Create required directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['predictions_dir'], exist_ok=True)
        os.makedirs(self.config['monitoring_dir'], exist_ok=True)

        # Set derived paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reference_data_path = os.path.join(self.config['output_dir'], f"reference_data_{self.timestamp}.csv")
        self.model_path = None

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Config file {config_path} not found. Using default configuration.")
                return

            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            # Update configuration with loaded values
            self.config.update(loaded_config)
            logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.warning("Falling back to default configuration")

    def _split_reference_data(self) -> None:
        """
        Split data into training and reference sets.
        """
        try:
            logger.info(f"Loading data from {self.config['data_file']}")
            df = pd.read_csv(self.config['data_file'])

            # Randomly sample a portion for reference data
            reference_size = int(len(df) * self.config['reference_data_ratio'])
            reference_data = df.sample(n=reference_size, random_state=42)

            # Save reference data
            reference_data.to_csv(self.reference_data_path, index=False)
            logger.info(f"Reference data saved to {self.reference_data_path}")

        except Exception as e:
            logger.error(f"Error splitting reference data: {str(e)}")
            raise

    def _run_subprocess(self, command: str) -> int:
        """
        Run a command as a subprocess.

        Args:
            command: Command to run

        Returns:
            Return code from the process
        """
        logger.info(f"Running command: {command}")

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Stream output in real-time
            for stdout_line in iter(process.stdout.readline, ""):
                print(stdout_line, end="")

            for stderr_line in iter(process.stderr.readline, ""):
                print(stderr_line, end="", file=sys.stderr)

            # Wait for process to complete
            return_code = process.wait()

            if return_code != 0:
                logger.error(f"Command failed with return code {return_code}")
            else:
                logger.info(f"Command completed successfully")

            return return_code

        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return 1

    def run_data_processing(self) -> int:
        """
        Run the data processing step.

        Returns:
            Return code from the process
        """
        command = f"python data_processor.py --input_file {self.config['data_file']} --output_file processed_data.csv"
        return self._run_subprocess(command)

    def train_model(self) -> int:
        """
        Run the model training step.

        Returns:
            Return code from the process
        """
        command = f"python model_trainer.py --data_file {self.config['data_file']} --output_dir {self.config['output_dir']}"
        return_code = self._run_subprocess(command)

        if return_code == 0:
            # Find the latest model file
            model_files = [f for f in os.listdir(self.config['output_dir']) if f.endswith('.joblib')]
            if model_files:
                latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.config['output_dir'], x)))
                self.model_path = os.path.join(self.config['output_dir'], latest_model)
                logger.info(f"Latest model found: {self.model_path}")
            else:
                logger.warning("No model file found after training")

        return return_code

    def generate_predictions(self) -> int:
        """
        Run the prediction step.

        Returns:
            Return code from the process
        """
        if not self.model_path:
            logger.error("No model path available. Cannot generate predictions.")
            return 1

        output_file = os.path.join(self.config['predictions_dir'], f"predictions_{self.timestamp}.csv")

        command = f"python predictor.py --model_path {self.model_path} --input_file {self.config['data_file']} --output_file {output_file}"
        return self._run_subprocess(command)

    def run_monitoring(self) -> int:
        """
        Run the model monitoring step.

        Returns:
            Return code from the process
        """
        if not self.model_path or not os.path.exists(self.reference_data_path):
            logger.error("Model path or reference data not available. Cannot run monitoring.")
            return 1

        email_param = f"--email {self.config['notification_email']}" if self.config['notification_email'] else ""

        command = (
            f"python monitor.py --model_path {self.model_path} "
            f"--reference_data {self.reference_data_path} "
            f"--new_data {self.config['data_file']} "
            f"--monitoring_dir {self.config['monitoring_dir']} "
            f"{email_param}"
        )
        return self._run_subprocess(command)

    def run_pipeline(self) -> None:
        """
        Run the complete MLOps pipeline.
        """
        try:
            logger.info("Starting MLOps pipeline")

            # Step 1: Split data for monitoring
            logger.info("Step 1: Splitting reference data")
            self._split_reference_data()

            # Step 2: Data processing
            logger.info("Step 2: Data processing")
            if self.run_data_processing() != 0:
                logger.error("Data processing failed. Stopping pipeline.")
                return

            # Step 3: Model training
            logger.info("Step 3: Model training")
            if self.train_model() != 0:
                logger.error("Model training failed. Stopping pipeline.")
                return

            # Step 4: Generate predictions
            logger.info("Step 4: Generating predictions")
            if self.generate_predictions() != 0:
                logger.error("Prediction generation failed. Stopping pipeline.")
                return

            # Step 5: Run monitoring
            logger.info("Step 5: Running model monitoring")
            if self.run_monitoring() != 0:
                logger.warning("Model monitoring completed with issues")

            logger.info("MLOps pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in MLOps pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RTV Risk Assessment MLOps Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--data_file', type=str, help='Path to the data file')
    parser.add_argument('--output_dir', type=str, help='Directory for model output')
    parser.add_argument('--skip_monitoring', action='store_true', help='Skip monitoring step')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MLOpsPipeline(args.config)

    # Override config with command line arguments if provided
    if args.data_file:
        pipeline.config['data_file'] = args.data_file

    if args.output_dir:
        pipeline.config['output_dir'] = args.output_dir

    # Run the pipeline
    try:
        if args.skip_monitoring:
            # Run only data processing, training and prediction
            logger.info("Starting MLOps pipeline (skipping monitoring)")
            pipeline._split_reference_data()
            pipeline.run_data_processing()
            pipeline.train_model()
            pipeline.generate_predictions()
        else:
            # Run the full pipeline
            pipeline.run_pipeline()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

    logger.info("Pipeline execution completed")