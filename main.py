import dataLoader
import dataProcessing
import eda
import outlierDetection
import regressionModel
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # Data Loading
    raw_df = dataLoader.merge_data("raw")
    raw_df.to_csv("raw_merged_data.csv", index=False)

    # Data Processing
    processed_df = dataProcessing.process_data(raw_df)
    processed_df.to_csv("processed_data.csv", index=False)

    # EDA
    # eda.perform_eda(processed_df)

    # Outlier Detection
    cleaned_df = outlierDetection.outliers(processed_df)
    cleaned_df.to_csv("cleaned_data.csv", index=False)

    # Regression Model
    model, predictions = regressionModel.regression_model(cleaned_df, target="demand_mwh", time_column="datetime")