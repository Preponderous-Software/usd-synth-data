import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "blender: needs a Blender binary ($USD_SYNTH_BLENDER)")


@pytest.fixture(scope="session")
def blender():
    """The Blender binary, or a skip. CI sets USD_SYNTH_REQUIRE_BLENDER=1 so a skip fails."""
    from usd_synth.render import RenderError, find_blender

    try:
        return find_blender()
    except RenderError as err:
        if os.environ.get("USD_SYNTH_REQUIRE_BLENDER") == "1":
            pytest.fail(f"Blender required but unavailable: {err}")
        pytest.skip(str(err))
