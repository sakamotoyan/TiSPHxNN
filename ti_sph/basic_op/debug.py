from ..globals import mode, Mode

def DEBUG(*args):
    global mode
    if mode == Mode.DEBUG:
        print('[DEBUG]', end=" ")
        print(*args)
        print('')