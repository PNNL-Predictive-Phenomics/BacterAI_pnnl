"""Unit tests for simulation types and enums."""
import pytest
from src.bacterai.sim.core import SimType, SimDirection


class TestSimType:
    """Test suite for SimType enum."""

    def test_sim_type_values(self):
        """Test that all SimType enum values are defined."""
        assert SimType.RANDOM.value == 0
        assert SimType.GREEDY.value == 1
        assert SimType.ROLLOUT.value == 2
        assert SimType.ROLLOUT_PROB.value == 3

    def test_sim_type_names(self):
        """Test that SimType names are accessible."""
        assert SimType.RANDOM.name == "RANDOM"
        assert SimType.GREEDY.name == "GREEDY"
        assert SimType.ROLLOUT.name == "ROLLOUT"
        assert SimType.ROLLOUT_PROB.name == "ROLLOUT_PROB"


class TestSimDirection:
    """Test suite for SimDirection enum."""

    def test_sim_direction_values(self):
        """Test that all SimDirection enum values are defined."""
        assert SimDirection.DOWN.value == 0
        assert SimDirection.UP.value == 1
        assert SimDirection.BOTH.value == 2

    def test_sim_direction_names(self):
        """Test that SimDirection names are accessible."""
        assert SimDirection.DOWN.name == "DOWN"
        assert SimDirection.UP.name == "UP"
        assert SimDirection.BOTH.name == "BOTH"

    def test_action_value_down(self):
        """Test action_value() for DOWN direction."""
        result = SimDirection.DOWN.action_value()
        assert result == 0

    def test_action_value_up(self):
        """Test action_value() for UP direction."""
        result = SimDirection.UP.action_value()
        assert result == 1

    def test_action_value_both(self):
        """Test action_value() for BOTH direction."""
        result = SimDirection.BOTH.action_value()
        assert result == 1  # BOTH returns 1 (same as UP)

    def test_target_value_down(self):
        """Test target_value() for DOWN direction."""
        result = SimDirection.DOWN.target_value()
        assert result == 1

    def test_target_value_up(self):
        """Test target_value() for UP direction."""
        result = SimDirection.UP.target_value()
        assert result == 0

    def test_target_value_both(self):
        """Test target_value() for BOTH direction."""
        result = SimDirection.BOTH.target_value()
        assert result == 0  # BOTH returns 0 (opposite of action_value)
