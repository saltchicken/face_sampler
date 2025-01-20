import argparse
from .face_detector import run_face_sampler, run_face_sampler_on_folder

def main():
    parser = argparse.ArgumentParser(
        description="Sample face from an image and force resolution to 512 X 512"
    )

    parser.add_argument("-i", "--input", required=True, help="Input image")
    parser.add_argument(
        "-o", "--output", default="output.png", type=str, help="Output folder"
    )
    parser.add_argument(
        "--folder", action="store_true", help="Converting folder of images to faces"
    )

    args = parser.parse_args()

    if args.folder:
        run_face_sampler_on_folder(args.input, args.output)
    else:
        run_face_sampler(args.input, args.output)
