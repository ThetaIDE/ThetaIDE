"""User-requested research activity on macOS, without privileged scheduling changes."""
from __future__ import annotations

import sys
from contextlib import contextmanager

_CURRENT = {}


def scheduling_metadata():
    return dict(_CURRENT)


@contextmanager
def research_activity():
    if sys.platform != "darwin":
        yield
        return
    import ctypes
    from contextlib import ExitStack

    details = {"requested": "user_initiated", "app_nap_disabled": False, "thread_qos_applied": False}
    with ExitStack() as cleanup:
        try:
            from Foundation import NSActivityUserInitiated, NSProcessInfo
            process = NSProcessInfo.processInfo()
            activity = process.beginActivityWithOptions_reason_(NSActivityUserInitiated, "Theta research experiment")
            cleanup.callback(process.endActivity_, activity)
            details["app_nap_disabled"] = True
        except Exception as exc:
            details["activity_error"] = str(exc)
        try:
            system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            system.pthread_self.restype = ctypes.c_void_p
            system.pthread_get_qos_class_np.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
            system.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
            previous, relative = ctypes.c_uint(), ctypes.c_int()
            error = system.pthread_get_qos_class_np(system.pthread_self(), ctypes.byref(previous), ctypes.byref(relative))
            if error:
                raise OSError(error, "Could not read worker QoS")
            # QOS_CLASS_USER_INITIATED from the macOS sys/qos.h SDK header.
            error = system.pthread_set_qos_class_self_np(0x19, 0)
            if error:
                raise OSError(error, "Could not set worker QoS")
            cleanup.callback(system.pthread_set_qos_class_self_np, previous.value, relative.value)
            details["thread_qos_applied"] = True
        except Exception as exc:
            details["qos_error"] = str(exc)
        _CURRENT.update(details)
        try:
            yield
        finally:
            _CURRENT.clear()
