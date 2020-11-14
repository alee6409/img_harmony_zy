# img_harmony_zy
## train yolov3 with your images.

_______________________________________________________________________________________
well i am finally back!
with my new codes(which combind many great team`s ideas). they are urtralytics(https://github.com/ultralytics/yolov3) and ChriswooTalent(https://github.com/ChriswooTalent/COCO_forYOLO). I cant finish my graduation thesis without their brillant codes. So thank u again(although u maybe wont see it ^_^)

now let me introduce this structure.
first of all, if u want to repeat this repository(which is easy), u must read ultralytics`s repository(especially this one: https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) and understand it.
that page contains the baseline of the whole procedure.
it tells u that if u want to create your custom data, this is a very strict pattern for the data. (AND THAT IS WHY I need ChriswooTalent's code, because I train my data in a COCO format, so i have to change it to a darknet format. and actually it is not a neccessary step for u if your format is already fits darknet's format.)

i put my old codes to the {old} directory and the {COCO_forYOLO} directory contains tools to change the coco format data to a darknet format.
the {yolov3} directory contains the whole tools to train your own data. Although the structure is really clear, i still pursuade u to read the {train.py} and the {detect.py} carefully. maybe u can modify them for your own uses.

at last, the {rgb2grey.py} is for changing the colorful images to black-white ones. if u dont need, just ignore it.
_______________________________________________________________________________________
well today i find a project from . it does the same thing as mine, 
however! it is awesome! really awesome! let me learn from them then update this resp.

----------------------------------------------------------------------------------------
+ detectzy.py is a detect file(run file)

+ darknetzy.py is yolov3 structure network

+ utilzy.py contains some useful tools for darknet and detection

+ run/darknet director contains something about tensorboard

+ data/*.names contains categories in datasets(if you create your own datasets, maybe you need to create one too)

+ cfg/yolov3.cfg is the official cfg file, which contains the structure of yolov3 darknet-53

adjust_hue.py and rgb2grey.py are my tools, which only create my datasets from COCO datasets and not neccessary to this framework.

***use it like***

```python3 detectzy.py --path2img 'your absolute path to img director' --path2json 'your absolute path to json file'```

1. only support **coco-like** images/json file now.
2. default reso is 608
3. due to time limited, up to now, the default batch_size is 1, maybe bigger in the future.

to be continued.

*reference*:
AYOOSH KATHURIA
https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
thank him a lot ^_^

l1119289003@outlook.com feel free to connect me if any bug flies hhh