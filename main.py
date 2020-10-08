import sys
from recognition import Recognition
from recognition_demo import RecognitionDemo
from reconstruction import Reconstruction


if __name__ == "__main__":

    # Recognize actions
    if(sys.argv[1] == "recognition"):
        r = Recognition(sys.argv[2:])
    elif(sys.argv[1] == "recognition_demo"):
        r = RecognitionDemo(sys.argv[2:])
    elif (sys.argv[1] == "reconstruction"):
        r = Reconstruction(sys.argv[2:])
    else:
        print("Please specify supported modes: recognition, reconstruction, prediction")
        exit(-1)

    r.start()
