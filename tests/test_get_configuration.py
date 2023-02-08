import os, sys
from src import get_configuration


def test_get_settings():
    config_file = 'src/config_main.ini'
    if not os.path.isfile(config_file):
        raise ValueError(f"\nConfig file, '{config_file}', does not exist!")
    cdict = get_configuration.get_settings(config_file)
    assert len(cdict.keys()) > 1
    assert type(cdict) is dict
