from CharacterDetection.test import BigLetterDetection
from CharacterClassification import test

cd = BigLetterDetection()
cd.detect('../OCR/story_image_1.jpg')

test.classify('../OCR/temp.jpg')