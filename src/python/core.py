import os
import ctypes

go = ctypes.cdll.LoadLibrary(os.environ["GICORE"])

go.echo.argtypes = [ctypes.c_char_p]
go.echo.restype = ctypes.c_void_p
def echo(name: str):
    ret = go.echo(name.encode("utf-8"))
    return ctypes.string_at(ret).decode('utf-8')
