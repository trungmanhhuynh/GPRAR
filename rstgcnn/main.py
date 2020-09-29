
import sys
from recognition import Recognition

if __name__ == "__main__":

    r = Recognition(sys.argv[1:])
    r.start()
