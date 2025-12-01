"""Integration tests for configuration loading and setup."""
import pytest
import json
from src.bacterai.run.loader import load_experiment_config, load_ingredients_data
from src.bacterai.run.setup import setup_experiment


class TestConfigurationLoading:
    """Integration tests for configuration and ingredient loading."""

    def test_load_experiment_config(self, setup_experiment_dir):
        """Test loading experiment configuration from JSON."""
        config = load_experiment_config(str(setup_experiment_dir))
        
        assert "grow_threshold" in config
        assert "batch_size" in config
        assert "experiment_path" in config
        assert config["batch_size"] == 96

    def test_load_experiment_config_missing(self, temp_experiment_dir):
        """Test loading config from directory without config.json."""
        with pytest.raises(FileNotFoundError):
            load_experiment_config(str(temp_experiment_dir))

    def test_load_ingredients_data(self, setup_experiment_dir, sample_config):
        """Test loading ingredients data."""
        config = load_experiment_config(str(setup_experiment_dir))
        ingredients_pd = load_ingredients_data(str(setup_experiment_dir), config)
        
        assert not ingredients_pd.empty
        assert "INGREDIENT" in ingredients_pd.columns
        assert "TYPE" in ingredients_pd.columns
        assert "MIN_VALUE" in ingredients_pd.columns
        assert "MAX_VALUE" in ingredients_pd.columns

    def test_load_ingredients_filters_grouped(self, setup_experiment_dir, sample_config):
        """Test that grouped reagents (with ::) are filtered out."""
        # Modify ingredients.json to include a grouped reagent
        config = load_experiment_config(str(setup_experiment_dir))
        ingredients_path = setup_experiment_dir / "ingredients.json"
        
        with open(ingredients_path, "r") as f:
            ingredients_json = json.load(f)
        
        # Add a grouped ingredient
        ingredients_json["ingredients"].append({
            "INGREDIENT": "group::reagent",
            "TYPE": "binary",
            "MIN_VALUE": 0.0,
            "MAX_VALUE": 1.0,
            "NOMINAL_VALUE": 1.0,
            "N_STATES": 2,
        })
        
        with open(ingredients_path, "w") as f:
            json.dump(ingredients_json, f)
        
        # Reload and check
        ingredients_pd = load_ingredients_data(str(setup_experiment_dir), config)
        assert "group::reagent" not in ingredients_pd["INGREDIENT"].tolist()

    def test_load_ingredients_filters_invariant(self, setup_experiment_dir, sample_config):
        """Test that invariant ingredients (MIN == MAX) are filtered out."""
        config = load_experiment_config(str(setup_experiment_dir))
        ingredients_path = setup_experiment_dir / "ingredients.json"
        
        with open(ingredients_path, "r") as f:
            ingredients_json = json.load(f)
        
        # Add an invariant ingredient
        ingredients_json["ingredients"].append({
            "INGREDIENT": "invariant_reagent",
            "TYPE": "quantitative",
            "MIN_VALUE": 5.0,
            "MAX_VALUE": 5.0,  # Same as MIN
            "NOMINAL_VALUE": 5.0,
            "N_STATES": 1,
        })
        
        with open(ingredients_path, "w") as f:
            json.dump(ingredients_json, f)
        
        # Reload and check
        ingredients_pd = load_ingredients_data(str(setup_experiment_dir), config)
        assert "invariant_reagent" not in ingredients_pd["INGREDIENT"].tolist()


class TestExperimentSetup:
    """Integration tests for full experiment setup."""

    def test_setup_experiment_basic(self, setup_experiment_dir):
        """Test basic experiment setup."""
        settings, ingredients_pd, ingredients_list = setup_experiment(str(setup_experiment_dir))
        
        # Check settings object
        assert hasattr(settings, "round_number")
        assert hasattr(settings, "grow_threshold")
        assert hasattr(settings, "batch_size")
        assert settings.batch_size == 96
        
        # Check ingredients
        assert not ingredients_pd.empty
        assert len(ingredients_list) == len(ingredients_pd)
        assert all(isinstance(name, str) for name in ingredients_list)

    def test_setup_experiment_sim_types(self, setup_experiment_dir):
        """Test that simulation types are properly converted."""
        settings, _, _ = setup_experiment(str(setup_experiment_dir))
        
        from src.bacterai.sim.core import SimType
        assert isinstance(settings.simulation_types, list)
        assert all(isinstance(st, SimType) for st in settings.simulation_types)

    def test_setup_experiment_direction(self, setup_experiment_dir):
        """Test that direction is properly converted."""
        settings, _, _ = setup_experiment(str(setup_experiment_dir))
        
        from src.bacterai.sim.core import SimDirection
        assert isinstance(settings.direction, SimDirection)

    def test_setup_experiment_model_type(self, setup_experiment_dir):
        """Test that model type is properly converted."""
        settings, _, _ = setup_experiment(str(setup_experiment_dir))
        
        from src.bacterai.run.models import ModelType
        assert isinstance(settings.model_type, ModelType)
