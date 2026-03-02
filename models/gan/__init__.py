from .discriminators import VGGDiscriminator, UNetDiscriminator
from .srgan import SRGAN
from .esrgan import ESRGAN
from .real_esrgan import RealESRGAN

__all__ = ["VGGDiscriminator", "UNetDiscriminator", "SRGAN", "ESRGAN", "RealESRGAN"]
