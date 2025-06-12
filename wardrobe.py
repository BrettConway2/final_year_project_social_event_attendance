
from feature import Feature


class Wardrobe():
    def __init__(self, hat: Feature, sunglasses: Feature, upper_clothes: Feature, skirt: Feature, pants: Feature, dress: Feature, belt: Feature, left_shoe: Feature, right_shoe: Feature, bag: Feature, scarf: Feature):
        self.hat = hat
        self.sunglasses = sunglasses
        self.upper_clothes = upper_clothes
        self.skirt = skirt
        self.pants = pants
        self.dress = dress
        self.belt = belt
        self.left_shoe = left_shoe
        self.right_shoe = right_shoe
        self.bag = bag
        self.scarf = scarf


        items = [hat, upper_clothes, skirt, pants, belt, bag, scarf]
        self.count = sum(1 for item in items if item is not None)

