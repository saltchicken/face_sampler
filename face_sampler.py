import cv2
import dlib
import argparse
from time import time
import matplotlib.pyplot as plt
from pyesrgan import run_esrgan
import os, sys
from pyframes import video_to_frames
import tempfile


def facecenter_squarecrop_image(image, face_center):
    height, width, _ = image.shape
    # TODO Check if already 512 x 512
    min_dimension = min(height, width)
    if height == min_dimension:
        left_offset = face_center['width']
        right_offset = width - face_center['width']
        # print('face_center_width', face_center['width'])
        # print('min_dimension', min_dimension)
        centered_min_dimension = min_dimension // 2
        if centered_min_dimension > left_offset:
            add_to_right = centered_min_dimension - left_offset
            # print('right_offset', face_center['width']+centered_min_dimension+add_to_right)
            cropped_image = image[0:height, 0:face_center['width']+centered_min_dimension+add_to_right]
        elif centered_min_dimension > right_offset:
            add_to_left = centered_min_dimension - right_offset
            # print('left_offset', face_center['width']-centered_min_dimension-add_to_left)
            cropped_image = image[0:height,face_center['width']-centered_min_dimension-add_to_left:width]
        else:
            # TODO Validate that the addition of 1 to the right_offset is necessary when the centered_min_dimension is odd which might mess with resizing. If issues this isn't necessary
            if centered_min_dimension % 2 == 0:
                cropped_image = image[0:height, face_center['width']-centered_min_dimension:face_center['width']+centered_min_dimension]
            else:
                cropped_image = image[0:height, face_center['width']-centered_min_dimension:face_center['width']+centered_min_dimension+1]
        # if face_center['width'] >= width / 2:
        #     cropped_image = image[0:height, width - min_dimension:width]
        #     # crop_width = [width - min_dimension : width]
        # else:
        #     cropped_image = image[0:height, 0:min_dimension]
        #     # crop_width = [0 : min_dimension]
        
    else:
        top_offset = face_center['height']
        bottom_offset = height - face_center['height']
        centered_min_dimension = min_dimension // 2
        if centered_min_dimension > top_offset:
            add_to_bottom = centered_min_dimension - top_offset
            cropped_image = image[0:face_center['height']+centered_min_dimension+add_to_bottom, 0:width]
        elif centered_min_dimension > bottom_offset:
            add_to_top = centered_min_dimension - bottom_offset
            cropped_image = image[face_center['height']-centered_min_dimension-add_to_top:height,0:width]
        else:
            # TODO Validate that the addition of 1 to the right_offset is necessary when the centered_min_dimension is odd which might mess with resizing. If issues this isn't necessary
            if centered_min_dimension % 2 == 0:
                cropped_image = image[face_center['height'] - centered_min_dimension:face_center['height']+centered_min_dimension]
            else:
                cropped_image = image[face_center['height'] - centered_min_dimension:face_center['height']+centered_min_dimension+1]
        # if face_center['height'] >= height / 2:
        #     cropped_image = image[height - min_dimension : height, 0:width]
        # else:
        #     cropped_image = image[0 : min_dimension, 0:width]
    return cropped_image
    
def run_face_sampler(input, output):
    image = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    face_center = hogDetectFaces(image)
    if face_center == False:
        print('no face found: skipping')
        return False
    cropped_image = facecenter_squarecrop_image(image, face_center)
    with tempfile.TemporaryDirectory() as temp_dir:
        cv2.imwrite(temp_dir + '/saved.png', cropped_image)
        height, width, _ = cropped_image.shape
        if height < 512:
            # TODO Pass cropped_image cv2 object directly to pyesrgan
            run_esrgan(temp_dir + '/saved.png', output, resolution=(512,512))
        else:
            resized_image = cv2.resize(cropped_image, (512,512))
            print('output', output)
            cv2.imwrite(output, resized_image)
            
# def video_to_frames(input, output, rate):
#     ffmpeg_command = [
#         'ffmpeg',
#         '-i', input,
#         '-r', rate,
#         output + '/frames' + '/%05d.png'
#     ]

#     # Run the ffmpeg command
#     try:
#         subprocess.run(ffmpeg_command, check=True)
#         print(f'Conversion complete.')
#     except subprocess.CalledProcessError as e:
#         print(f'Error during conversion: {e}')

def run_face_sampler_on_video(input, output, rate):
    with tempfile.TemporaryDirectory() as temp_dir:
        video_to_frames(input, temp_dir, rate)
        run_face_sampler_on_folder(temp_dir, output)
        
def run_face_sampler_on_folder(input_path, output_folder):
    print('Current working directory', os.getcwd())
    if os.path.exists(output_folder):
        print(f"The directory {output_folder} exists.")
        # TODO: Check if folder is not empty and if so throw an error
    else:
        print(f"The directory '{output_folder}' does not exist.")
        os.mkdir(output_folder)
    all_items = os.listdir(input_path)
    file_names = [item for item in all_items if os.path.isfile(os.path.join(input_path, item))]
    for file_name in file_names:
        run_face_sampler(input_path + '/' + file_name, output_folder + '/' + file_name)
    
def main():
    parser = argparse.ArgumentParser(description="Sample face from an image and force resolution to 512 X 512")
    
    parser.add_argument('-i', '--input', required=True, help='Input image')
    parser.add_argument('-o', '--output', default='output.png', type=str, help='Output folder')
    parser.add_argument('-r', '--rate', default='10', type=str, help='Framerate for video conversion')
    parser.add_argument('--video', action='store_true', help='Converting video to faces')
    parser.add_argument('--folder', action='store_true', help='Converting folder of images to faces')
    
    args = parser.parse_args()
    
    if args.video and args.folder:
        print("Can't specify both video and folder")
        sys.exit(1)
    elif args.video:
        run_face_sampler_on_video(args.input, args.output, args.rate)
    elif args.folder:
        run_face_sampler_on_folder(args.input, args.output)
    else:
        run_face_sampler(args.input, args.output)
    
    
    
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

    # TODO modify this for multiple faces. Right now it simply returning the last one it finds discarding others.
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
    if len(results) > 0:    
        return {"width":xCenter, "height":yCenter}
    else:
        return False

if __name__ == '__main__':
    main()