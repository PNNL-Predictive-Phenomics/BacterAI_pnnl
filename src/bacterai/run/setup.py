from .paths import get_round
from .loader import load_experiment_config, load_ingredients_data
from .models import ModelType
from ..sim.core import SimType, SimDirection


class Settings:
	"""Simple namespace object for settings."""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)


def setup_experiment(experiment_path: str):
	"""Setup experiment configuration and data."""
	round_number = get_round(experiment_path)
	config = load_experiment_config(experiment_path)
    
	# Create settings dict with defaults
	settings_dict = {
		'round_number': round_number,
		'grow_threshold': config["grow_threshold"],
		'experiment_path': experiment_path,  # Use the passed-in path, not the config value
		'nickname': f"{config['nickname']}R{round_number}",
		'batch_size': config["batch_size"],
		'timeout_min': config["timeout_min"],
		'n_rollouts': config["n_rollouts"],
		'model_type': ModelType(config["model_type"]),
		'direction': SimDirection(config["direction"]),
		'simulation_types': [SimType(x) for x in config["simulation_types"]],
		'beyond_frontier': config["beyond_frontier"],
		'use_unique': config["use_unique"],
		'ingredients_file': config.get("ingredients_file"),
		'transfer_model_folder': config.get("transfer_model_folder"),
		'redo_size': config.get("redo_size"),
		'redo_threshold': config.get("redo_threshold"),
		'aas_only': config.get("aas_only", False),
		'transfer_data_dir': config.get("transfer_model_dir"),
		'separate_redos': config.get("separate_redos", False),
		'n_bags': config.get("n_bags", 25),
		'random_walk_increment': config.get("random_walk_increment", 10),
	}
    
	ingredients_pd = load_ingredients_data(experiment_path, config)
	ingredients_list = ingredients_pd["INGREDIENT"].tolist()
    
	return Settings(**settings_dict), ingredients_pd, ingredients_list