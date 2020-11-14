from PIL import Image
import glob
import os
import time



if __name__=='__main__':
    
    path = 'd:/cs/data/coco/train2017/bw/'

    for filename in glob.glob(os.path.join(path, '*.jpg')):

        img = Image.open('{:s}'.format(filename)).convert('L')
        img.save('d:/cs/data/coco/train2017/hue_bw/grey_{:s}'.format(filename[29:]))