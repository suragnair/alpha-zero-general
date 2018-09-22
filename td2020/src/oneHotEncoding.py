from typing import List


def itb(num: int, length: int) -> List[int]:
    if length == 2:
        return [int(i) for i in '{0:02b}'.format(num)]
    if length == 4:
        return [int(i) for i in '{0:04b}'.format(num)]
    if length == 5:
        return [int(i) for i in '{0:05b}'.format(num)]
    if length == 13:
        return [int(i) for i in '{0:013b}'.format(num)]
    raise TypeError("Length not supported")


def bti(num: List[int]) -> int:
    return int("".join([str(i) for i in num]), 2)

# print(bti(itb(6,13)))
