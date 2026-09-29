import torch
import matplotlib.pyplot as plt

samples = torch.load("outputs/samples2.pt", map_location="cpu")
images = (samples +1)/2
fig, axs = plt.subplots(4,4,figsize=(8, 8))
fig.suptitle("Generated MNIST Digits")
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i].squeeze(),cmap="gray")
    ax.axis("off")
plt.savefig("outputs/samples2.png")
plt.show()