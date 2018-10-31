import cv2
import sys
from src import input
from src import extract
from src import match
from src import stitch

def main(Limg_name, Rimg_name):
    
    Limg = cv2.imread(Limg_name)
    Rimg = cv2.imread(Rimg_name)

    Limg = input.equalize_histogram_color(Limg)
    Rimg = input.equalize_histogram_color(Rimg)

    Lgray = cv2.cvtColor(Limg,cv2.COLOR_BGR2GRAY)
    Rgray = cv2.cvtColor(Rimg,cv2.COLOR_BGR2GRAY)

    Lfeatures = extract.get_features(Lgray)
    Rfeatures = extract.get_features(Rgray)

    matches = match.get_matches(Lgray, Rgray, Lfeatures, Rfeatures)

    result = stitch.stitch(matches, Limg, Rimg, Lfeatures, Rfeatures)

    cv2.imshow("RESULT", result)
    cv2.imshow("LEFT", Limg)
    cv2.imshow("RIGHT", Rimg)
    #
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])