
import sys
from recognition import Recognition
from reconstruction import Reconstruction


if __name__ == "__main__":

    # Recognize actions
    if(sys.argv[1] == "recognition"):
        r = Recognition(sys.argv[2:])
        r.start()
    elif (sys.argv[1] == "reconstruction"):
        r = Reconstruction(sys.argv[2:])
        r.start()
    else:
        print("Please specify supported modes: recognition, reconstruction, prediction")
    # Reconstruct poses

    # Prediction
