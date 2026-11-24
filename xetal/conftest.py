"""Pytest settings for the X_eTaL libraries' tests (run from the repository root)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the model extra and the weights")
