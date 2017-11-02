# -*- coding: utf-8 -*-

import os
import copy

Annnotation = """<annotation>
	<folder>DIAN2017</folder>
	<filename>{}</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
        {}
</annotation>
"""
Object = """
	<object>
		<name>{}</name>
		<pose>Left</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
"""

def write2xml(filename, height, width, cls_name, bboxes):
    """Save class name and bbox into a xml file

    Args:
        filename (str): xml filename
        height (int): image heigth
        width (int): image width
        cls_name (str): class name
        bboxes (list[int]): [[x1, y1, x2, y2],...]

    Returns: null

    """

    objs = ""
    for name, bbox in zip(cls_name, bboxes):
        assert(len(bbox) == 4)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        xmin = max(0, xmin) + 1
        xmax = min(width, xmax)
        ymin = max(0, ymin) + 1 
        ymax = min(height, ymax)

        newObj = copy.deepcopy(Object).format(name, xmin, ymin, xmax, ymax)
        objs += newObj
    newAnno = copy.deepcopy(Annnotation).format(os.path.basename(filename)[:-4], width, height, objs)

    xmlfile = filename
    if os.path.exists(xmlfile):
        os.remove(xmlfile)
    with open(xmlfile, 'w') as fid:
        fid.write(newAnno)
