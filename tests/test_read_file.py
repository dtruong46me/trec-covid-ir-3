import pandas as pd

def test_read_file(filepath: str):
    """
    Reads a file and returns its contents. Supports CSV and TXT formats.
    Args:
        filepath (str): The path to the file to be read.
    Returns:
        pd.DataFrame or list: The contents of the file as a DataFrame (for CSV) or a list (for TXT).
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        return df
    elif filepath.endswith(".txt"):
        with open(filepath, "r") as file:
            content = file.readlines()
        return [line.strip() for line in content]
    else:
        raise ValueError("Unsupported file format.")

if __name__ == "__main__":
    FILEPATHS = {
        "corpus": "data/metadata.csv",
        "queries": "data/topics-rnd3.csv",
        "ground_truth": "data/qrels.csv",
        # "access": "data/cord_19_embeddings_2025-05-19.csv",
        "sample_output": "data/sample_submission.csv",
        "filter": "data/docids-rnd3.txt",
    }

    results = {}

    for key, path in FILEPATHS.items():
        results[key] = test_read_file(path)
        print(f"Successfully read {key} from {path}")

    print("All files read successfully.")