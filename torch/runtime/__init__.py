from collections import deque
import torch

_current_runtime = 'cpu'

_run_time_dict = {}


def _available_runtimes():
    print(_run_time_dict)


def register_runtime(name, module):
    """Register a runtime module for backends to expose the APIs not being included in ATen."""
    global _current_runtime

    if name in _run_time_dict:
        raise RuntimeError("Runtime {} already registered with {}.".format(name, _run_time_dict[name]))
    if hasattr(torch, name):
        raise RuntimeError("torch attribute {} already exists.".format(name))
    try:
        # the runtime type should have the same name as the device type
        device = torch.device(name)
    except RuntimeError as e:
        raise RuntimeError("Unsupported runtime type {}.".format(name)) from e


    attributes = ['current_device', 'device_count', 'get_device_properties', 'is_available']

    def check_attributes(attr):
        if not hasattr(module, attr):
            raise RuntimeError("There is no {} in runtime {}.".format(attr, name))

    for attr in attributes:
        check_attributes(attr)

    _run_time_dict[name] = module
    setattr(torch, name, module)
    _current_runtime = name


def current_runtime():
    global _current_runtime
    return _run_time_dict[_current_runtime]


def get_device_type() -> str:
    return _current_runtime


def get_runtime_attr(get_member):
    runtime = current_runtime()
    return get_member(runtime)
