import os
import shutil
import argparse
image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}

def parse_args():
    parser = argparse.ArgumentParser(description="Put all places 365 images in single folder.")
    parser.add_argument("-i",
                        "--input_dir",
                        required=True,
                        type=str,
                        help="input folder: the folder containing unzipped places 365 files")
    parser.add_argument("-o",
                        "--output_dir",
                        required=True,
                        type=str,
                        help="output folder: the folder to put all images")
    parser.add_argument("-u",
                        "--unzip",
                        required=False,
                        default=False,
                        type=str,
                        help="unzip if need")
    args = parser.parse_args()
    return args

def genlist(image_dir):
    image_list = []
    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir,filename)
        if os.path.isdir(path):
            image_list = image_list + genlist(path)
        else:
            ext = os.path.splitext(filename)[1]
            if ext in image_extensions:
                image_list.append(os.path.join(image_dir, filename))
    return image_list


args = parse_args()
if bool(args.unzip):
    import zipfile
    with zipfile.ZipFile(args.input_dir, 'r') as zip_ref:
        zip_ref.extractall(args.output_dir)
    print('----unzip done----')
else:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    flist = genlist(args.input_dir)
    for i,p in enumerate(flist):
        if os.path.getsize(p) != 0:
            os.rename(p,os.path.join(args.output_dir,str(i)+'.jpg'))
    shutil.rmtree(args.input_dir)
    print('----done----')