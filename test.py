import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from cyclegan import Generator

def process_large_image(large_image_path, pytorch_model, output_path):
    large_image = Image.open(large_image_path)

    width, height = large_image.size
    num_tiles_x = (width + 255) // 256
    num_tiles_y = (height + 255) // 256

    processed_image = Image.new('RGB', (num_tiles_x * 256, num_tiles_y * 256))

    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            tile = large_image.crop((x * 256, y * 256, (x + 1) * 256, (y + 1) * 256))
            
            tile_tensor = TF.to_tensor(tile).unsqueeze(0)

            processed_tile_tensor = pytorch_model(tile_tensor)

            processed_tile = TF.to_pil_image(processed_tile_tensor.squeeze(0))

            processed_image.paste(processed_tile, (x * 256, y * 256))

    processed_image.save(output_path)

input_channels = 3
model = Generator(input_channels)
state_dict = torch.load('./model_checkpoints/G_BA_epoch_99.pth')
model.load_state_dict(state_dict)
model.eval()
tile_tensor = TF.to_tensor(Image.open('test.jpg').resize((256, 256))).unsqueeze(0)
processed_tile_tensor = model(tile_tensor)
processed_tile = TF.to_pil_image(processed_tile_tensor.squeeze(0))
processed_tile.save('out.jpg')
# process_large_image('test.jpg', model, 'out.jpg')