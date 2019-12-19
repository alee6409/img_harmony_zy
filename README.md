# img_harmony_zy
## train yolov3 with your images.

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
3. up to now, the default batch_size is 1, maybe bigger in the future. I haven`t gotton time to do this

to be continued.

*reference*:
AYOOSH KATHURIA
https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
thanks him a lot ^_^

l1119289003@outlook.com feel free to connect me if any bug flies hhh