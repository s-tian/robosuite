import glob
import sys
import os
import h5py
import robosuite
import datetime
import copy


def gather_demonstrations_as_hdf5(directory, out_dir):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print(hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0

    for ep_directory in os.listdir(directory):

        h5_path = os.path.join(directory, ep_directory, "demo.hdf5")
        print(h5_path)
        f_orig = h5py.File(h5_path, 'r')

        orig_eps = 1
        while True:
            try:
                f_orig[f'data/demo_{orig_eps}']
            except KeyError as e:
                break
            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            orig = f_orig[f'data/demo_{orig_eps}']
            ep_data_grp.attrs["model_file"] = copy.deepcopy(orig.attrs['model_file'])

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=orig["states"][:].copy())
            ep_data_grp.create_dataset("actions", data=orig["actions"][:].copy())
            orig_eps += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = copy.deepcopy(f_orig['data'].attrs["repository_version"])
    grp.attrs["env"] = copy.deepcopy(f_orig['data'].attrs["env"])
    grp.attrs["env_info"] = copy.deepcopy(f_orig['data'].attrs["env_info"])

    f.close()


if __name__ == '__main__':
    gather_demonstrations_as_hdf5(sys.argv[1], sys.argv[2])