from omegaconf import OmegaConf

def load_config(config_file='config.yaml'):
    config = OmegaConf.load(config_file)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    return config