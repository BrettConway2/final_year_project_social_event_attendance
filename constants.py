MTCNN_FACE_CONFIDENCE_THRESHOLD = 0.95 # Min. confidence for face to be detected (within model)

# figure_detector.py
FIGURE_DETECTION_THRESHOLD = 0.65 # Min. confidence for a figure to be detected (within model)
CLOTHING_DETECTION_THRESHOLD = 0.5

# Main.py
FIGURE_MATCH_THRESHOLD = 0.55 # max distance between 2 figures (using figure.likeness) to allow a match                ######## 0.5 -> 0.55
FIGURE_PIXEL_QUANTITY_THRESHOLD = 200
FACE_MATCH_THRESHOLD = 0.90   # max between 2 faces to match
FACE_DIFFERENTIATION_THRESHOLD = 1.0 # Max facial distance (for detected faces) to tolerate a match on non-facial data ########  1.2 -> 1.3

LACK_OF_KNOWLEDGE_PENALTY = 150

KMEANS_CLUSTERS = 5 # 2 for foreground, 3 for bg

MIN_MASK_SIZE = 500 # pixels

NN_INPUT_DIM = 20


######### NN HYPERPARAMETERS

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 80

EMBEDDING_BOTH_MISSING_DEFAULT = 3.5
EMBEDDING_ONE_MISSING_DEFAULT = 5.0

PALETTE_BOTH_MISSING_DEFAULT = 7.0
PALETTE_ONE_MISSING_DEFAULT = 7.0
