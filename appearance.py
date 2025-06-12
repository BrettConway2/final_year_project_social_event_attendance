from figure import Figure


class Appearance():
    def __init__(self, figure: Figure, photo_num: int, match_type: str="", test_label = ""):
        self.figure = figure
        self.photo_num = photo_num
        self.match_type = match_type
        self.test_label = test_label
    
    def __eq__(self, other):
        if not isinstance(other, Appearance):
            return False
        return self.photo_num == other.photo_num and self.figure == other.figure and self.match_type == other.match_type
