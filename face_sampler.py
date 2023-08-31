import cv2
import dlib
import argparse
from time import time
import matplotlib.pyplot as plt
from pyesrgan import run_esrgan
import tempfile


def facecenter_squarecrop_image(image, face_center):
    height, width, _ = image.shape
    min_dimension = min(height, width)
    if height == min_dimension:
        if face_center['width'] >= width / 2:
            cropped_image = image[0:height, width - min_dimension:width]
            # crop_width = [width - min_dimension : width]
        else:
            cropped_image = image[0:height, 0:min_dimension]
            # crop_width = [0 : min_dimension]
        
    else:
        if face_center['height'] >= height / 2:
            cropped_image = image[height - min_dimension : height, 0:width]
        else:
            cropped_image = image[0 : min_dimension, 0:width]
    return cropped_image
    

def main():
    parser = argparse.ArgumentParser(description="Automatically setup EBSynth.")
    
    parser.add_argument('-i', '--input', required=True, help='Input image')
    parser.add_argument('-o', '--output', default='output', type=str, help='Output folder')
    
    args = parser.parse_args()
    
    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    face_center = hogDetectFaces(image)
    cropped_image = facecenter_squarecrop_image(image, face_center)
    with tempfile.TemporaryDirectory() as temp_dir:
        cv2.imwrite(temp_dir + '/saved.png', cropped_image)
        height, width, _ = cropped_image.shape
        if height < 512:
            # TODO Pass cropped_image cv2 object directly to pyesrgan
            run_esrgan(temp_dir + '/saved.png', 'resized.png', resolution=(512,512))
        else:
            resized_image = cv2.resize(cropped_image, (512,512))
            cv2.imwrite('resized.png', resized_image)
    
    # output_image = cv2.imread('resized.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow('output', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
            
def hogDetectFaces(image):
    hog_face_detector = dlib.get_frontal_face_detector()
    
    # output_image = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hog_face_detector(imgRGB, 0)
    # print(results)

    for bbox in results:
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()
        # cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=width//200)
        xCenter = int((x1 + x2) / 2) 
        yCenter = int((y1 + y2) / 2)
        # cv2.circle(output_image, (xCenter, yCenter), radius=3, color=(0,0,255), thickness=5)
        # cv2.rectangle(output_image, pt1=(xCenter - 256, yCenter + 256), pt2=(xCenter + 256, yCenter - 256), color=(0, 255, 0), thickness=width//200)
        
    return {"width":xCenter, "height":yCenter}

if __name__ == '__main__':
    main()