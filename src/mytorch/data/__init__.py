from ..utils.importer import import_method


def get_dataset(cfg):
    data_method = import_method(cfg.MODULE)
    dataset = data_method()
    return dataset
