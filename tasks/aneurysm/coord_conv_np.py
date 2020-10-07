# -*- coding=utf-8 -*-
import numpy as np


class AddCoordsNp():
    """Add coords to a tensor"""

    def __init__(self, rank, with_r=False):
        super().__init__()
        assert rank in [1, 2, 3], 'invalid param, rank={}'.format(rank)
        self.rank = rank
        self.with_r = with_r

    def __call__(self, input_tensor):
        """
        input_tensor: (b, c, ...)
        """
        if self.rank == 1:
            pass

        elif self.rank == 2:
            pass

        elif self.rank == 3:
            pass

        else:
            raise NotImplementedError

        return input_tensor

class AddCoordsNp4GCP():
    """ Add coords to a tensor
    (Add global coords to patch tensor) """

    def __init__(self, rank, size, with_r=False):
        """
        :param rank: the axes count of tensor
        :param size: full size of each axes(full image tensor)
        :param with_r:
        """
        super().__init__()
        assert rank in [1, 2, 3], 'invalid param, rank={}'.format(rank)
        self.rank = rank
        self.size = size
        self.with_r = with_r

    def __call__(self, patch_tensor, offset=0):
        """
        patch_tensor: (b, c, ...), patch
        offset: coords offset, default value: 0
        """
        if isinstance(offset, int):
            offset = (offset,) * self.rank
        if self.rank == 1:
            pass

        elif self.rank == 2:
            pass

        elif self.rank == 3:
            pass
        else:
            raise NotImplementedError

        return patch_tensor

if __name__ == '__main__':
    # addcoords = AddCoordsNp(rank=3, with_r=False)
    # input_tensor = np.zeros((2, 1, 3, 4, 5))
    # out = addcoords(input_tensor)
    # print(out[0])

    input_tensor = np.zeros((1, 1, 30, 40, 50))
    addcoords = AddCoordsNp(rank=3, with_r=False)
    out1 = addcoords(input_tensor)

    addcoords = AddCoordsNp4GCP(rank=3, size=(30, 40, 50), with_r=False)
    patch_tensor = input_tensor[..., 10:20, 11:22, 12:24]
    out2 = addcoords(patch_tensor, offset=(10, 11, 12))

    out12 = out1[..., 10:20, 11:22, 12:24]
    print(np.any(out12 == out2))

    # from IPython import embed
    # embed()
