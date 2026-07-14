from prefect import flow, task

from src.data.load_data import process_and_save_data
from src.models.train_model import train_model
from src.monitoring.drift_report import generate_drift_report


@task
def process_data_task():
    return process_and_save_data()


@task
def train_model_task():
    train_model()


@task
def generate_drift_report_task():
    generate_drift_report()


@flow(name="credit-risk-training-pipeline")
def training_pipeline():
    process_data_task()
    train_model_task()
    generate_drift_report_task()


if __name__ == "__main__":
    training_pipeline()