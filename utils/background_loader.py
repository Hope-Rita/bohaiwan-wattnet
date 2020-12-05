from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class BackGroundLoader(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
