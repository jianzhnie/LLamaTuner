from typing import Sequence

from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor


def _preprocess_visual_inputs(images: Sequence[ImageObject],
                              processor: ProcessorMixin) -> 'NDArray':
    # process visual inputs (currently only supports a single image)
    image_processor: BaseImageProcessor = getattr(processor, 'image_processor')
    image = (images[0] if len(images) != 0 else Image.new(
        'RGB', (100, 100), (255, 255, 255)))
    return image_processor(image, return_tensors='pt')['pixel_values'][0]
