import numpy as np
from ColourSelection import rgb_to_name
from appearance import Appearance


class Person():
    def __init__(self, appearances: list[Appearance] = []):
        self.appearances: list[Appearance] = appearances
        self.centroid = None
        self.face_embedding = None
        self.description = ""

        if len(appearances) > 0 and self.face_embedding is None and appearances[0].figure.face_data is not None:
            self.face_embedding =  appearances[0].figure.face_data.embedding

    def __eq__(self, other):
        if not isinstance(other, Person):
            return False
        return self.appearances == other.appearances
    
    def update_centroid(self):
        faces = [a.figure.face_data for a in self.appearances if a.figure.face_data is not None]
        if len(faces) > 0:
            self.face_embedding = faces[0].embedding
        
        item_counts = [a.figure.count for a in self.appearances]
        self.centroid = self.appearances[np.argmax(item_counts)].figure
    
    def add_appearance(self, appearance):
        self.appearances.append(appearance)
        if self.face_embedding is None and appearance.figure.face_data.embedding is not None:
            self.face_embedding =  appearance.figure.face_data.embedding




    def get_description(self):
        self.update_centroid()
        attribute_labels = ["hair", "skin", "shirt/dress", "pants", "skirt", "scarf", "bag", "hat", "belt"]

        attributes = [[a.figure.hair,
                      a.figure.body,
                      a.figure.wardobe.upper_clothes,
                      a.figure.wardobe.pants,
                      a.figure.wardobe.skirt,
                      a.figure.wardobe.scarf,
                      a.figure.wardobe.bag,
                      a.figure.wardobe.hat,
                      a.figure.wardobe.belt] for a in self.appearances]
        
        attributes_by_column = list(zip(*(attributes.copy())))

        # Count non-None entries for each attribute
        non_none_counts = [sum(1 for item in column if item is not None) for column in attributes_by_column]
        reliable_attr = [count >= len(self.appearances) / 2 for count in non_none_counts]

        centroid_attrs = [self.centroid.hair,
                          self.centroid.body,
                          self.centroid.wardobe.upper_clothes,
                          self.centroid.wardobe.pants,
                          self.centroid.wardobe.skirt,
                          self.centroid.wardobe.scarf,
                          self.centroid.wardobe.bag,
                          self.centroid.wardobe.hat,
                          self.centroid.wardobe.belt]

        
        description = []
        
        for n, centroid_attr in enumerate(centroid_attrs):
            if centroid_attr is not None and centroid_attr.colour is not None and reliable_attr[n]:
                label = attribute_labels[n]
                colour_text = rgb_to_name(centroid_attr.colour)
                description.append(colour_text + " " + label)

        self.description = ', '.join(description)

        return description

        
