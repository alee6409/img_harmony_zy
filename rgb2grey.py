from PIL import Image
import glob
import os
import time

if __name__=='__main__':
    
    
    path = '/Users/zhiyilee/cs/python/yolov3_zy'

    for filename in glob.glob(os.path.join(path, '*.png')):

        img = Image.open(filename[36:]).convert('LA')
        img.save('grey{:s}'.format(filename[36:]))