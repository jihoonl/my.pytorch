from ..utils.importer import import_method


def get_dataset(cfg):
    data_method = import_method(cfg.MODULE)
    dataset = data_method(cfg)
    return dataset


def get_preprocessed_data(data, cfg):
    preprocess = import_method(cfg.PREPROCESS)
    return preprocess(data)
