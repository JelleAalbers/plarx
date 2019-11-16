import random
import string


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


def random_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))