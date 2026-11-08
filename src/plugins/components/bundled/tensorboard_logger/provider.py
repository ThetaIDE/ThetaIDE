"""TensorBoard execution adapter; optional Lightning dependency loads on invocation."""
def create_logger(settings, context):
    from lightning.pytorch.loggers import TensorBoardLogger
    log_dir = context.base_root / 'results/tensorboard' / context.group_name / context.experiment_id
    logger = TensorBoardLogger(str(log_dir), name=context.method_name, default_hp_metric=False)
    _ = logger.experiment
    return logger
