from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):
    """
    A custom Image folder that returns in addition to images, the path of each image. This class is used for
    visualization purposes.

    """
    def __getitem__(self, index):
        """
        Redefinition of getitem method.

        :return: a custom ImageFolder containing images paths
        """
        # Getting original output (images and labels)
        original_output = super().__getitem__(index)

        # Getting images paths
        paths = self.imgs[index][0]

        return original_output + (paths,)
