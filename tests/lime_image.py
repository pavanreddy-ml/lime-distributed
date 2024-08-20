import sys
import os

# Set workdir to Root of project
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


from lime.lime_image import LimeImageExplainer, LimeImageExplainerDistributed
import numpy as np
import time
import random
import asyncio


li = LimeImageExplainerDistributed(num_workers=1)
random_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


async def pred_async(images):
    print("predicting")
    await asyncio.sleep(random.randint(2, 3))
    batch_size = images.shape[0]
    return np.random.randint(0, 10, batch_size)


async def main():
    await li.explain_instance(random_image, pred_async, batch_size=128, num_samples=1000000)

asyncio.run(main())