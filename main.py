import dataLoader
import dataProcessing
import eda

if __name__ == "__main__":

    # Data Loading
    raw_df = dataLoader.merge_data("raw")
    raw_df.to_csv("raw_merged_data.csv", index=False)

    # Data Processing
    processed_df = dataProcessing.process_data(raw_df)
    processed_df.to_csv("processed_data.csv", index=False)

    # EDA
    eda.perform_eda(processed_df)