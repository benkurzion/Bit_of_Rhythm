import cv2
from main import paper_algorithm
import numpy as np

def apply_ellipse_to_drumstick_head(img):
    img= cv2.resize(img, (225,225))
    print(img.shape)
    final_clusters = paper_algorithm(img)
    for cluster in final_clusters:
        if len(cluster) >= 100:
            ellipse = cv2.fitEllipse(np.array(cluster))
            cv2.ellipse(img, ellipse, (0,255,0), 2)

    cv2.imshow("Drum Head", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    im_path = "./drumsticks.jpg"
    image = cv2.imread(im_path)
    apply_ellipse_to_drumstick_head(image)


