from pathlib import Path

import yaml


def parse_yaml(file_path):
    try:
        config_path = Path(file_path).resolve()

        if not config_path.is_file():
            raise FileNotFoundError(f"can't find {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            base_path = []

            if not config.get('FL'):
                return config

            if not config.get('FL').get('dataset'):
                raise ValueError("can't find FL.dataset in config file")

            base_path1 = Path('config/_base_') / f"{config['FL']['dataset']}.yaml"
            base_path.append(base_path1.resolve())

            if not config.get('FL').get('sampler'):
                raise ValueError("can't find FL.sampler in config file")

            base_path2 = Path('config/sampler') / f"{config['FL']['sampler']}.yaml"
            base_path.append(base_path2.resolve())

            if config['FL']['with_DP']:
                base_path3 = Path('config/dp') / f"{config['FL']['dp']}.yaml"
                base_path.append(base_path3.resolve())
            if config['FL']['with_attack']:
                base_path4 = Path('config/attack') / f"{config['FL']['attack']}.yaml"
                base_path.append(base_path4)
            if config['FL']['with_defense']:
                base_path5 = Path('config/defense') / f"{config['FL']['defense']}.yaml"
                base_path.append(base_path5)
            if config['FL']['with_recover']:
                base_path6 = Path('config/recover') / f"{config['FL']['recover']}.yaml"
                base_path.append(base_path6)

            for path in base_path:
                if not path.is_file():
                    raise FileNotFoundError(f"can't find {path}")
                base_config = parse_yaml(path)
                config = {**base_config, **config}

            return config

    except yaml.YAMLError as e:
        raise ValueError(f"YAML PARSE ERROR: {str(e)}") from e
