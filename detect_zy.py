from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 as cv

import argparse
import os 
import os.path as osp
from darknet_zy import Darknet
from util_zy import *
import pickle as pkl
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import json

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epoches during training")   
    # parser.add_argument("--images", dest = 'images', help = 
    #                     "Image / Directory containing images to perform detection upon",
    #                     default = "imgs", type = str)
    # parser.add_argument("--det", dest = 'det', help = 
    #                     "Image / Directory to store detections to",
    #                     default = "det", type = str)
    parser.add_argument("--batch_size", type=int, help = "Batch size", default = 1)
    parser.add_argument("--confidence", type=float, help = "Object Confidence to filter predictions", default = 1)
    parser.add_argument("--nms_thresh", type=float, help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    # parser.add_argument("--weights", dest = 'weightsfile', help = 
    #                     "weightsfile",
    #                     default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "608", type = str)
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--cpu', type = int, default=2, help='number of cup cores')
    parser.add_argument('--train/val', type = str, default='train', help='train or val mode')
    parser.add_argument('--num_anchors', type=int, default=3, help='number of anchors')
    return parser.parse_args()
    
args = arg_parse()


CUDA = torch.cuda.is_available()

# Configure data loader

#os.makedirs("../../data/coco/train2017/test_example", exist_ok=True)
darknet_datasets = datasets.CocoDetection(root='d:/cs/data/coco/train2017_2/', 
                                          annFile = 'd:/cs/data/coco/annotations/instances_train2017.json',
                                          transform = transforms.Compose([transforms.ToTensor(), 
                                                                          transforms.Normalize([0.5], [0.5])]))
# dataloader = torch.utils.data.DataLoader(
#     darknet_datasets, batch_size = args.batch_size, shuffle = True,
# )

##need to get annotations
#/////annotations = dataloader

num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")

#initialize network
darknet = Darknet(args.cfgfile)

#training by ourselves
#model.load_weights(args.weightsfile)
#print("Network successfully loaded")


#loss function
Darknet_loss = torch.nn.BCELoss()

#writer
writer = SummaryWriter('run/darknet')

darknet.net_info["height"] = args.reso
inp_dim = int(darknet.net_info["height"])
# assert inp_dim % 32 == 0 
# assert inp_dim > 32


if CUDA:
    Darknet_loss.cuda()
    darknet.cuda()

#optimizer
optimizer_G = torch.optim.Adam(darknet.parameters(), lr=args.lr, betas=(args.b1, args.b2))



#read annotations from *.json files
#json->"annotation"->many->"bbox", "category_id", "is_crowded"
#change category_id to tuple(np.zeros(80))

            


#Set the model in evaluation mode
#model.eval()

# read_dir = time.time()
# #Detection phase
# try:
#     imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
# except NotADirectoryError:
#     imlist = []
#     imlist.append(osp.join(osp.realpath('.'), images))
# except FileNotFoundError:
#     print ("No file or directory with the name {}".format(images))
#     exit()
    
# if not os.path.exists(args.det):
#     os.makedirs(args.det)

# load_batch = time.time()
# ori_loaded_ims = [cv.imread(x) for x in imlist]

# im_batches = list(map(prep_image, ori_loaded_ims, [inp_dim for x in range(len(imlist))]))
# ori_im_dim = [(x.shape[1], x.shape[0]) for x in ori_loaded_ims]
# ori_im_dim = torch.FloatTensor(ori_im_dim).repeat(1, 2)


# leftover = 0
# if (len(ori_im_dim) % batch_size):
#     leftover = 1

# if batch_size != 1:
#     num_batches = len(imlist) // batch_size + leftover            
#     im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
#                         len(im_batches))]))  for i in range(num_batches)]  

# write = 0


# if CUDA:
#     ori_im_dim = ori_im_dim.cuda()
    
# start_det_loop = time.time()


######################training#######################

train_start = time.time()
for epoch in range(args.n_epochs):
    epoch_start = time.time()

    for i, (img, anno) in enumerate(darknet_datasets):

        
        #load the image 
        img_start = time.time()

        img = img.numpy().transpose(1, 2, 0)
        img = torch.from_numpy(img)
        img = prep_image(img, inp_dim)

        
        if CUDA:
            img = img.cuda()
            

        prediction = darknet(Variable(img), CUDA)
        

        #nms_conf
        prediction = cal_conf(prediction, anno, args.confidence, CUDA, args.batch_size)



        valid = read_anno(anno)
        
        if CUDA:
            valid = valid.cuda()
        darknet_loss = Darknet_loss(torch.sigmoid(prediction), torch.sigmoid(valid))

        darknet_loss.requires_grad = True

        optimizer_G.zero_grad()

        darknet_loss.backward()
        optimizer_G.step()

        #save weights
        torch.save(darknet.state_dict(), 'd:/cs/python/img_harmony/yolov3_zy.weights')

        img_end = time.time()
        img_time = img_end - img_start
        print('img time:{:f}'.format(img_time))

    epoch_end = time.time()
    epoch_time = epoch_start - epoch_end
    print('epoch time:{:d}'.format(epoch_time))

train_end = time.time()
train_time = train_end - train_start
print('train time:{:d}'.format(train_time))
        #load weights
        #darknet = Darknet(*args, **kwargs)
        #darknet.load_state_dict(torch.load('/Users/zhiyilee/cs/python/img_harmony'))
        #model.eval()


    
#     prediction = write_results(prediction, confidence, num_classes, nms_thesh)

#     #no detections
#     if type(prediction) == int:

#         for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
#             im_id = i*batch_size + im_num
#             print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#             print("{0:20s} {1:s}".format("Objects Detected:", ""))
#             print("----------------------------------------------------------")
#         continue

#     prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

#     if not write:                      #If we have't initialised output
#         output = prediction  
#         write = 1
#     else:
#         output = torch.cat((output,prediction))

#     for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
#         im_id = i*batch_size + im_num
#         objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
#         print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#         print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
#         print("----------------------------------------------------------")

#     if CUDA:
#         torch.cuda.synchronize()



# try:
#     output
# except NameError:
#     print ("No detections were made")
#     exit()



# ori_im_dim = torch.index_select(ori_im_dim, 0, output[:,0].long())

# scaling_factor = torch.min(608/ori_im_dim,1)[0].view(-1,1)

# #rescale coordinates
# output[:,[1,3]] -= (inp_dim - scaling_factor*ori_im_dim[:,0].view(-1,1))/2
# output[:,[2,4]] -= (inp_dim - scaling_factor*ori_im_dim[:,1].view(-1,1))/2



# output[:,1:5] /= scaling_factor

# for i in range(output.shape[0]):
#     output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, ori_im_dim[i,0])
#     output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, ori_im_dim[i,1])
    
    
# output_recast = time.time()
# class_load = time.time()
# colors = pkl.load(open("pallete.bin", "rb"))

# draw = time.time()


# def write(x, results):
#     c1 = tuple(x[1:3].int())
#     c2 = tuple(x[3:5].int())
#     img = results[int(x[0])]

#     cls = int(x[-1])
#     color = random.choice(colors)
#     label = "{0}".format(classes[cls])
#     cv.rectangle(img, c1, c2,color, 5)

#     t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1 , 1)[0]
#     c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#     cv.rectangle(img, c1, c2,color, -1)
    
#     cv.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
#     return img


# list(map(lambda x: write(x, ori_loaded_ims), output))

# det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

# list(map(cv.imwrite, det_names, ori_loaded_ims))


# end = time.time()

# print("SUMMARY")
# print("----------------------------------------------------------")
# print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
# print()
# print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
# print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
# print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
# print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
# print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
# print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
# print("----------------------------------------------------------")


# torch.cuda.empty_cache()
    
