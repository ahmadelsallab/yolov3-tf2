import xml.etree.ElementTree as ET
from os import getcwd
import os
import argparse

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#VOC_dir = '../VOCdevkit/'
def convert_annotation(year, image_id, list_file, VOC_dir):
    #in_file = open('../VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open(VOC_dir + '/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
def _main(args):
    VOC_dir = args.VOC_dir
    for year, image_set in sets:
        #image_ids = open('../VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        image_ids = open(VOC_dir + 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        file_name = VOC_dir + '%s_%s.txt'%(year, image_set)
        list_file = open(file_name, 'w')
        #file_name = '%s_%s.txt'%(year, image_set)
        #list_file = open(file_name, 'w')
        #if not os.path.isfile(file_name):
        for image_id in image_ids:
            #list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(VOC_dir, year, image_id))
            #list_file.write('/content/drive/My\ Drive/Colab\ Notebooks/Practical_DL_ITI_2019_CV/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
            convert_annotation(year, image_id, list_file, VOC_dir)
            list_file.write('\n')
        list_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='YOLO v3 Custom Training')
    parser.add_argument('--VOC_dir', help='the path to the VOC data', default='VOCdevkit/')
    args = parser.parse_args()




    _main(args)