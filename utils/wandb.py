import torch
import wandb

from metrics.distance_metrics import DistanceEvaluation
from models.classifier import Classifier
from utils.training_config_parser import TrainingConfigParser


def load_model(run_path,
               model_path=None,
               config=None,
               architecture=None,
               num_classes=None,
               replace=True):

    # Get file path at wandb if not set
    if model_path is None:
        api = wandb.Api(timeout=60)
        run = api.run(run_path)
        model_path = run.config["model_path"]
        architecture = run.config['Architecture']

    # Create model
    if num_classes is None:
        num_classes = run.config["num_classes"]

    if config:
        model = config.create_model()
    elif architecture is None:
        architecture = model_path.split('/')[-1].split('_')[0]

    model = Classifier(num_classes, in_channels=3, architecture=architecture)

    # Load weights from wandb
    file_model = wandb.restore(model_path,
                               run_path=run_path,
                               root='./weights',
                               replace=replace)

    # Load weights from local file
    model.load_state_dict(
        torch.load(file_model.name, map_location='cpu')['model_state_dict'])

    model.wandb_name = run.name

    return model


def load_config(run_path, config_name):
    config_file = wandb.restore(config_name,
                                run_path=run_path,
                                root='./configs',
                                replace=True)
    with open(config_file.name, "r") as config:
        config = TrainingConfigParser(config)
    return config
