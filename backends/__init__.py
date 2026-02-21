def detect_backend(config):
    if 'backend' in config:
        return config['backend']
    if 'model_config' in config:
        return 'sd'
    model_path = config.get('pretrained_model_name_or_path', '')
    if 'flux' in model_path.lower():
        return 'flux'
    return 'sd'
