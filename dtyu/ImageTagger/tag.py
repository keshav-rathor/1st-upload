""" a rip of tag_processing notebook for tag info access
    added: LabelSelector for generating a specific attribute label
"""

import os
import xml.etree.ElementTree as ET
from enum import Enum


ImageFeatures = {}
MainImageFeatures = {}
AllTags = {}
SimulatedFeatures = {}


class tagtype(Enum):
    Real = 0
    Synthetic = 1


# XML tag parser
class tagdata:
    def __init__(self, path, imagedir='', prefix=None, tag_type=tagtype.Real):
        # init
        self.xml = None
        self.name = ''
        self.path = path
        self.ImageDir = imagedir
        self.ImagePath = ''
        self.ImageFeatures = []
        self.MainImageFeatures = []
        self.SimulatedFeatures = []
        self.tags = []
        self.tag_type = tag_type
        if path is not None:
            self.load(path, imagedir=imagedir, prefix=prefix, tag_type=tag_type)

    def load(self, path, imagedir, prefix=None, tag_type=tagtype.Real):
        """load and parse an xml tag record.
            args: path  path of the xml file
                imagedir    path of the corresponding image file (may be unused)
                prefix  prefix bit of the absolute path of the sims data (to be removed when storing the path)
                tag_type    Real or Synthetic
            output: selected tag properties + the raw xml ElementTree
        """
        self.xml = ET.parse(path)
        root = self.xml.getroot()
        feature = None
        if tag_type == tagtype.Real:
            self.name = root.attrib['path'] + root.attrib['filename']
            self.ImagePath = os.path.join(imagedir, self.name)  # the use of ImagePath may be discontinued to decouple tag and image loading
            for tag in root.iter('tag'):
                self.tags.append(tag.text)
                if not tag.text in AllTags:
                    AllTags[tag.text] = 1
                else:
                    AllTags[tag.text] = AllTags[tag.text] + 1
            for feat in root.findall('primary_category'):
                if feat.attrib['name'] == 'Scattering features in image':
                    feature = feat
                    break
            if feature is not None:
                # row_tag contains major image feature labels (eg. halo, peak, etc.)
                for row_tag in feature.iter('row_tag'):
                    mainfeat = row_tag.attrib['name']
                    self.MainImageFeatures.append(mainfeat)
                    if not mainfeat in MainImageFeatures:
                        MainImageFeatures[mainfeat] = 1
                    else:
                        MainImageFeatures[mainfeat] = MainImageFeatures[mainfeat] + 1
                for tag in feature.iter('tag'):
                    self.ImageFeatures.append(tag.text)
                    if not tag.text in ImageFeatures:
                        ImageFeatures[tag.text] = 1
                    else:
                        ImageFeatures[tag.text] = ImageFeatures[tag.text] + 1
        elif tag_type == tagtype.Synthetic:
            for feat in root.findall('protocol'):
                if feat.attrib['name'] == 'tag_generated':
                    feature = feat
                    break
            if feature is not None:
                self.name = feature.attrib['outfile']
                if prefix is not None:
                    self.name = self.name.replace(prefix, '')   # to get the related image, /analysis/results/... still needs to be taken care of
                for tag in feature.iter('result'):
                    if tag.attrib['value'] == 'True':
                        simfeat = tag.attrib['name']
                        self.SimulatedFeatures.append(simfeat)
                        if not simfeat in SimulatedFeatures:
                            SimulatedFeatures[simfeat] = 1
                        else:
                            SimulatedFeatures[simfeat] += 1
        else:
            raise ValueError('Unrecognized tag type')

    def contains(self, tag):
        if self.tag_type == tagtype.Real:
            return tag in self.tags
        elif self.tag_type == tagtype.Synthetic:
            return tag in self.SimulatedFeatures
        else:
            raise ValueError('Unrecognized tag type')

    def add(self, tag):
        if self.tag_type == tagtype.Real:
            pass
        elif self.tag_type == tagtype.Synthetic:
            if not self.contains(tag):
                root = self.xml.getroot()
                feat = [elem for elem in root.findall('protocol')
                        if elem.attrib['name'] == 'tag_generated'][0]
                feat.append(ET.Element('result', {'name': tag, 'value': 'True'}))
                self.SimulatedFeatures.append(tag)
            else:
                raise MismatchedAttribute(tag + ' already existed')
        else:
            raise ValueError('Unrecognized tag type')

    def remove(self, tag):
        if self.tag_type == tagtype.Real:
            pass
        elif self.tag_type == tagtype.Synthetic:
            if self.contains(tag):
                root = self.xml.getroot()
                feat = [elem for elem in root.findall('protocol')
                        if elem.attrib['name'] == 'tag_generated'][0]
                tag_to_remove = [t for t in feat.iter('result')
                                 if t.attrib['name'] == tag][0]
                feat.remove(tag_to_remove)
                self.SimulatedFeatures.remove(tag)
            else:
                raise MismatchedAttribute(tag + ' does not exist')
        else:
            raise ValueError('Unrecognized tag type')


class MismatchedAttribute(Exception):
    pass
