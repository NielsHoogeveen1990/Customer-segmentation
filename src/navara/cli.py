import click
import click_pathlib
import logging
from navara.training import model_utils_train
from navara.create_segmentation import create_segmentations

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
    pass


@main.command()
@click.option("--data-path", type=click_pathlib.Path(exists=True))
@click.option("--model-version", type=int)
def train_model(data_path, model_version):
    model_utils_train.train(data_path, model_version)
    logger.info('Finished with training the model.')


@main.command()
@click.option("--input-path", type=click_pathlib.Path(exists=True))
@click.option("--output-path", type=click_pathlib.Path(exists=True))
def get_results(input_path, output_path):
    create_segmentations(input_path, output_path)
    logger.info('Finished with writing segmentation results.')