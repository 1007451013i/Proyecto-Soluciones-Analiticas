import pandas as pd
from sklearn.model_selection import train_test_split

from model.config.core import config
from model.pipeline import heart_pipe
from model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model and persist the pipeline."""

    # read training data
    data = load_dataset(file_name=config.app_config.train_data_file)

    # determine feature set
    features = config.model_config.features or [c for c in data.columns if c != config.model_config.target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data[features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
        stratify=data[config.model_config.target],
    )

    # fit model
    heart_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=heart_pipe)


if __name__ == "__main__":
    run_training()
