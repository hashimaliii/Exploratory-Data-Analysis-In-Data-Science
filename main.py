import dataLoader
import dataProcessing

if __name__ == "__main__":
    df = dataLoader.merge_data("raw")
    df.to_csv("raw_merged_data.csv", index=False)
    df = dataProcessing.process_data(df)
    df.to_csv("processed_data.csv", index=False)