import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
from typing import Tuple, Dict, List, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('data_processor')

class DataProcessor:
    """
    Handles data loading, cleaning, feature engineering, and validation
    for the RTV Risk Assessment Model.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataProcessor with configuration settings.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.scaler = StandardScaler()

        # Define demographic features
        self.demographic_cols = [
            'hhh_sex', 'hhh_age', 'hhh_educ_level', 'hhh_read_write',
            'tot_hhmembers', 'hh_size_adjusted'
        ]

        # Define income source features
        self.income_cols = [
            'business_number', 'work_salaried', 'work_casual',
            'borrowed_past_12_months', 'Rental_Income_Categories_1'
        ]

        # Define agricultural features
        self.agriculture_cols = [
            'Season1_cropped', 'Season1_land', 'Season2_cropped', 'Season2_land',
            'livestock_present_hh'
        ]

        # Define asset features
        self.asset_cols = [
            'farm_implements_owned', 'bicycle_owned', 'solar_owned', 'phones_owned'
        ]

        # Define expenditure features
        self.expenditure_cols = [
            'cereals_week', 'tubers_week', 'pulses_week', 'milk_week'
        ]

        # All base features
        self.base_features = (
            self.demographic_cols +
            self.income_cols +
            self.agriculture_cols +
            self.asset_cols +
            self.expenditure_cols
        )

        # Target variable
        self.target_column = 'HH Income + Production/Day (USD)'

        # Threshold for at-risk classification (in USD)
        self.risk_threshold = 2.0

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame containing the loaded data
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data for quality issues.

        Args:
            df: Input DataFrame

        Returns:
            Tuple containing (is_valid, list_of_issues)
        """
        issues = []

        # Check if required columns exist
        missing_cols = [col for col in self.base_features if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        # Check data types
        if 'hhh_age' in df.columns and not pd.api.types.is_numeric_dtype(df['hhh_age']):
            issues.append("Column 'hhh_age' is not numeric")

        # Check value ranges
        if 'hhh_age' in df.columns:
            invalid_ages = ((df['hhh_age'] < 15) | (df['hhh_age'] > 120)).sum()
            if invalid_ages > 0:
                issues.append(f"Found {invalid_ages} records with invalid ages")

        # Check row count
        if len(df) == 0:
            issues.append("Empty DataFrame, no records to process")

        # Check for extreme missing values
        missing_pct = df[self.base_features].isnull().mean()
        high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing_cols:
            issues.append(f"Columns with >50% missing values: {high_missing_cols}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean input data by handling missing values, outliers, etc.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle date columns first - convert to numeric features
        date_cols = []
        for col in df_clean.columns:
            # Try to identify date columns
            if df_clean[col].dtype == 'object':
                try:
                    # Check if column contains dates by trying to convert a sample
                    sample = df_clean[col].dropna().iloc[0] if not df_clean[col].dropna().empty else None
                    if sample and isinstance(sample, str):
                        if any(date_format in sample.lower() for date_format in ['am', 'pm', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                            date_cols.append(col)
                            try:
                                # Try to convert to datetime
                                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                                # Extract useful numeric features
                                df_clean[f'{col}_year'] = df_clean[col].dt.year
                                df_clean[f'{col}_month'] = df_clean[col].dt.month
                                df_clean[f'{col}_day'] = df_clean[col].dt.day
                                # Drop the original date column
                                df_clean = df_clean.drop(columns=[col])
                                logger.info(f"Converted date column '{col}' to numeric features")
                            except Exception as e:
                                logger.warning(f"Failed to convert date column '{col}': {str(e)}")
                                # If conversion fails, drop the column
                                df_clean = df_clean.drop(columns=[col])
                except Exception:
                    pass

        # Handle missing values for numeric columns
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col in df_clean.columns:
                # Replace missing values with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

                # Handle infinity values
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Handle missing values for categorical columns
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col in df_clean.columns:
                # Replace missing with mode
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")

        # Handle outliers for specific columns (using capping)
        outlier_cols = [col for col in ['hhh_age', 'Season1_land', 'Season2_land'] if col in df_clean.columns]
        for col in outlier_cols:
            Q1 = df_clean[col].quantile(0.05)
            Q3 = df_clean[col].quantile(0.95)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        logger.info(f"Data cleaning completed. Shape after cleaning: {df_clean.shape}")
        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional engineered features
        """
        # Create a copy to avoid modifying the original
        df_engineered = df.copy()

        # Land per person ratio
        if 'Season1_land' in df_engineered.columns and 'tot_hhmembers' in df_engineered.columns:
            df_engineered['land_per_person'] = df_engineered['Season1_land'] / df_engineered['tot_hhmembers'].replace(0, 1)

        # Food expense per person
        food_cols = [col for col in ['cereals_week', 'tubers_week', 'pulses_week', 'milk_week']
                   if col in df_engineered.columns]

        if food_cols and 'tot_hhmembers' in df_engineered.columns:
            df_engineered['food_expense_per_person'] = df_engineered[food_cols].sum(axis=1) / df_engineered['tot_hhmembers'].replace(0, 1)

        # Agricultural productivity
        if 'Season1_cropped' in df_engineered.columns and 'Season1_land' in df_engineered.columns:
            df_engineered['crop_utilization_ratio'] = df_engineered['Season1_cropped'] / df_engineered['Season1_land'].replace(0, np.nan)
            df_engineered['crop_utilization_ratio'] = df_engineered['crop_utilization_ratio'].fillna(
                df_engineered['crop_utilization_ratio'].median() if not pd.isna(df_engineered['crop_utilization_ratio'].median()) else 0
            )

        # Asset index
        asset_columns = [col for col in ['farm_implements_owned', 'bicycle_owned', 'solar_owned', 'phones_owned']
                       if col in df_engineered.columns]

        if asset_columns:
            df_engineered['asset_index'] = df_engineered[asset_columns].fillna(0).sum(axis=1)

        # Education factor
        if 'hhh_educ_level' in df_engineered.columns and 'hhh_read_write' in df_engineered.columns:
            df_engineered['education_factor'] = df_engineered['hhh_educ_level'] * df_engineered['hhh_read_write']

        # Income diversity
        income_source_columns = [col for col in ['business_number', 'work_salaried', 'work_casual', 'livestock_present_hh']
                               if col in df_engineered.columns]

        if income_source_columns:
            df_engineered['income_diversity'] = df_engineered[income_source_columns].replace(np.nan, 0).apply(
                lambda row: sum(row > 0), axis=1
            )

        # Log transform for skewed numeric variables
        skewed_cols = [col for col in ['cereals_week', 'tubers_week', 'pulses_week', 'milk_week', 'Season1_land', 'Season2_land']
                     if col in df_engineered.columns]

        for col in skewed_cols:
            if df_engineered[col].min() >= 0:  # Only transform non-negative columns
                df_engineered[f'{col}_log'] = np.log1p(df_engineered[col])

        logger.info(f"Feature engineering completed. New features added: {df_engineered.shape[1] - df.shape[1]}")

        # Get all features including engineered ones
        all_features = list(set(self.base_features) & set(df_engineered.columns))
        all_features.extend([col for col in df_engineered.columns if col not in self.base_features
                           and col not in [self.target_column, 'risk_status']])

        self.all_features = all_features

        return df_engineered

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable based on income threshold.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added target variable
        """
        df_target = df.copy()

        if self.target_column in df_target.columns:
            # Create binary target: 0 = at risk (< $2/day), 1 = not at risk (>= $2/day)
            df_target['risk_status'] = (df_target[self.target_column] >= self.risk_threshold).astype(int)

            logger.info(f"Target variable created. Distribution: "
                       f"At risk: {(df_target['risk_status'] == 0).sum()}, "
                       f"Not at risk: {(df_target['risk_status'] == 1).sum()}")
        else:
            logger.warning(f"Target column '{self.target_column}' not found. Target not created.")

        return df_target

    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Extract features and target from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple containing (features_df, target_series)
        """
        # Use all features we have available
        features = [col for col in self.all_features if col in df.columns]
        
        X = df[features].copy()
        
        # Ensure all features are numeric
        numeric_X = X.select_dtypes(include=['number'])
        
        # Log any dropped non-numeric columns
        dropped_cols = set(X.columns) - set(numeric_X.columns)
        if dropped_cols:
            logger.warning(f"Dropped non-numeric columns: {dropped_cols}")
        
        # Return target if it exists
        y = df['risk_status'] if 'risk_status' in df.columns else None
        
        return numeric_X, y

    def fit_scaler(self, X: pd.DataFrame) -> None:
        """
        Fit the scaler to the feature matrix.

        Args:
            X: Feature matrix
        """
        self.scaler.fit(X)
        logger.info("Fitted scaler to feature matrix")

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply scaling transformation to features.

        Args:
            X: Feature matrix

        Returns:
            Scaled feature matrix as numpy array
        """
        return self.scaler.transform(X)

    def process_data(self, file_path: str, for_training: bool = True) -> Tuple[np.ndarray, Optional[pd.Series]]:
        """
        Full data processing pipeline from loading to scaling.

        Args:
            file_path: Path to data file
            for_training: Whether the data is for training (affects scaler fitting)

        Returns:
            Tuple containing (scaled_features, target_series)
        """
        # Load data
        df = self.load_data(file_path)

        # Validate data
        is_valid, issues = self.validate_data(df)
        if not is_valid:
            logger.warning(f"Data validation found issues: {issues}")
            logger.info("Proceeding with processing despite validation issues")

        # Clean data
        df_clean = self.clean_data(df)

        # Engineer features
        df_engineered = self.engineer_features(df_clean)

        # Create target if needed
        if for_training:
            df_engineered = self.create_target(df_engineered)

        # Get features and target
        X, y = self.get_features_and_target(df_engineered)

        # Fit or apply scaling
        if for_training:
            self.fit_scaler(X)

        X_scaled = self.transform_features(X)

        return X_scaled, y

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Process data for RTV Risk Assessment Model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_file', type=str, help='Path to output processed data (optional)')

    args = parser.parse_args()

    processor = DataProcessor()

    try:
        X_scaled, y = processor.process_data(args.input_file)

        logger.info(f"Processed data shape: Features {X_scaled.shape}")
        if y is not None:
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        if args.output_file:
            # Convert back to DataFrame for saving
            processed_df = pd.DataFrame(X_scaled, columns=processor.all_features)
            if y is not None:
                processed_df['risk_status'] = y

            processed_df.to_csv(args.output_file, index=False)
            logger.info(f"Saved processed data to {args.output_file}")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")