




import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
from torchvision.transforms import CenterCrop


from detcon.models import Custom_Transform

image_path = "/gpfs/work5/0/prjs0790/data/random_pics/Screenshot 2024-03-20 at 18.30.52.png"
image = mpimg.imread(image_path)


plt.imshow(image)
plt.show()


img1 = torch.from_numpy(image)

img2 = img1[:, :, :3].permute(2, 0, 1)

plt.imshow(img2)
plt.show()



cropping = CenterCrop(size=224)

img2_cropped = cropping(img2)

img2_cropped.permute(1, 2, 0)

plt.imshow(img2_cropped.permute(1, 2, 0))
plt.show()

plt.imshow(img2)
plt.show()


ct = Custom_Transform(224)

out1 = ct(img2_cropped)
out2 = ct(img2_cropped)


plt.imshow(out1.permute(1, 2, 0))
plt.show()

plt.imshow(out2.permute(1, 2, 0))
plt.show()




img2_cropped
