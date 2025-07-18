import logging
import logging.config


def setup_logger(log_file=None, name='main_logger'):
    """create and set shared logger"""
    logger = logging.getLogger(name)

    # set the minimum recording level
    logger.setLevel(logging.DEBUG)

    # make sure that main_fed can set the saving path of log
    if log_file is None:
        return logger

    # prevent duplicate addition of handlers
    if logger.handlers:
        return logger

    # create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.DEBUG)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # set logger formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 禁用 root logger(吐槽opacus)
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'root': {
            'handlers': [],  # 不设置任何handlers
            'level': 'CRITICAL',  # 设置日志级别为CRITICAL，也可以设置为'NOTSET'来完全禁用
            'propagate': False  # 防止日志消息传递到上级logger
        }
    }

    # 应用日志配置
    logging.config.dictConfig(log_config)

    return logger
