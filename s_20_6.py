# Snippet 20.6 The nestedParts Function
import numpy as np


def NestedParts(numAtoms, numThreads, upperTriang=False):
    """Partition of atoms with an inner loop

    $$r_{m} = \frac{-1 + \sqrt{1 + 4 (r_{m-1}^{2} + r_{m-1} + N(N+1)M^{-1})}}{2}$$

    """
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are the heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def main():
    print(NestedParts(100, 16, False))


if __name__ == '__main__':
    main()
