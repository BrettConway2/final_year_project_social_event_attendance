from backend.feature import Feature


class ItemOfClothing(Feature):
    def __init__(self, image, item_type, colour, centre, mask, colour_palette, w, h, embedding):
        super().__init__(image, colour, centre, mask, colour_palette, w, h, embedding=embedding)
        self.item_type = item_type
