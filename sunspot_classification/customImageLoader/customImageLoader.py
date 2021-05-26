from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):

    # Redifinition of __getitem__
    def __getitem__(self, index):
        # Getting original output (images and labels)
        original_output = super().__getitem__(index)

        # Getting images paths
        paths = self.imgs[index][0]

        return original_output + (paths,)
