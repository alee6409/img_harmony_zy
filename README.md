# img_harmony_zy
## train yolov3 with your images.

+ detectzy.py is a detect file(run file)

+ darknetzy.py is yolov3 structure network

+ utilzy.py contains some useful tools for darknet and detection

adjust_hue.py and rgb2grey.py are my tools, which only create my datasets from COCO datasets and not neccessary to this framework.

***use it like***
```python3 detectzy.py --path2img 'your absolute path to img director' --path2json 'your absolute path to json director'```

1. only support **coco-like** images/json file now.
2. default reso is 608
3. up to now, the default batch_size is 1, maybe bigger in the future. I haven`t gotton time to do this

to be continued.

reference:
AYOOSH KATHURIA
https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
thanks him a lot ^_^