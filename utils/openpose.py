import sys
from sys import platform
import os


class OpenPose:

    __dir_path = os.path.dirname(os.path.realpath(__file__))
    __sys_path = __dir_path + '/../openpose/build/python/openpose/Release'
    __dll_path = __dir_path + '/../openpose/build/x64/Release'
    model_path = __dir_path + "/../openpose/models/"

    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(__sys_path)
            os.add_dll_directory(__dll_path)
            import pyopenpose
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../openpose/build/python/openpose')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the
            # OpenPose/python module from there. This will install OpenPose and the python library at your desired
            # installation path. Ensure that this is in your python path in order to use it. sys.path.append(
            # '/usr/local/python')
            import pyopenpose
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?')
        raise e

    op = pyopenpose


if __name__ == "__main__":
    op = OpenPose.op
    print(op)
    print(OpenPose.model_path)
