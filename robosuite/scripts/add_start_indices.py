import sys
import h5py


def add_indices(h5_file, labels):
    with open(labels, 'r') as f:
        lines = f.readlines()
    indices = list(map(int, lines))

    f = h5py.File(h5_file, "a") # core prevents MP collision, but should just load in at once?
    num_elements = len(list(f["data"].keys()))
    for i in range(1, num_elements+1):
        if f'data/demo_{i}/start_index' in f:
            f[f'data/demo_{i}/start_index'][...] = indices[i-1]
        else:
            f[f'data/demo_{i}/start_index'] = indices[i - 1]
        print(indices[i-1])
    f.close()


if __name__ == '__main__':
    add_indices(sys.argv[1], sys.argv[2])


