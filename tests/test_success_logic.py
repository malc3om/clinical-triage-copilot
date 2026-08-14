import pytest
def determine_success(rewards, done):
    if not done or not rewards: return False
    if any(r <= -5.0 for r in rewards): return False
    if sum(rewards) < 0.1: return False
    return True

def test_clean_run(): assert determine_success([0.10, 0.20, 0.30], True) is True
def test_fatal_kills_success(): assert determine_success([0.10, -10.0], True) is False
def test_early_positive_then_fatal(): assert determine_success([0.20, -10.0], True) is False
def test_below_threshold(): assert determine_success([0.02, 0.01], True) is False
def test_not_done(): assert determine_success([0.50, 0.30], False) is False
def test_empty(): assert determine_success([], True) is False
def test_safety_penalty_not_fatal(): assert determine_success([0.40, -0.50, 0.50], True) is True
