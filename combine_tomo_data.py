import sys
import os
import h5py
from typing import List, Tuple
from pathlib import Path

import numpy as np


XRF_DETECTOR = 'xsp3_addetector-xrf_windows-xsp3_addetector1'
# Assuming XRF data for now, so data key is templated due to the different
# elemental map data needing to be combined
DATA_KEY_TEMPLATE = lambda element: f"/processed/auxiliary/0-XRF Elemental Maps from ROIs/{element}/data"
PROJ_IMG_KEY = 0


def main(args):
    nxs_dir_path = Path(args[0])
    start_scan_no = int(args[1])
    end_scan_no = int(args[2])
    element_label = args[3]
    sample_desc = args[4]
    out_file_path = args[5]

    # Check if the given directory exists
    if not nxs_dir_path.exists():
        err_str = \
            f"The folder {nxs_dir_path} doesn't exist, please check your input"
        raise ValueError(err_str)

    filename_template = lambda scan_no: f"i14-{scan_no}-{XRF_DETECTOR}.nxs"
    nxs_file_paths = [Path(nxs_dir_path, filename_template(scan_no))
                      for scan_no in range(start_scan_no, end_scan_no+1)]
    
    nxs_files_angles = []
    # Check that all these files exist, and keep fetching the associated
    # rotation angle from them as long as they do exist
    for file_path in nxs_file_paths:
        if not file_path.exists():
            err_str = f"The file {file_path} doesn't exist."
            raise ValueError(err_str)
        else:
            angle = _get_rotation_angle(file_path)
            nxs_files_angles.append((file_path, angle))
    
    print('All required files exist!')

    # Sort the nxs files in ascending order based on their associated rotation
    # angle
    nxs_files_angles.sort(key=lambda tup: tup[1])

    print('The NeXuS files have been sorted!')

    max_x_dim, max_y_dim = _get_proj_max_dims(nxs_file_paths,
                                              DATA_KEY_TEMPLATE(element_label))
    print('Combining data...')
    combined_data, pad_info = \
        _combine_proj_data(nxs_files_angles, DATA_KEY_TEMPLATE(element_label),
                           max_x_dim, max_y_dim)
    img_keys = np.full(len(nxs_files_angles), PROJ_IMG_KEY)
    print('Saving file...')
    _write_combined_proj_data(combined_data,
                              [angle for (path, angle) in nxs_files_angles],
                              img_keys, sample_desc, out_file_path)
    print('Done!')


def _get_rotation_angle(file_path: Path,
                        angle_key: str='/entry/instrument/sample/sample_rot'
                        ) -> float:
    """Get the rotation angle associated with the given NeXuS file.

    Paramaters
    ----------
    file_path : Path
        The file path of the NeXuS file.
    
    angle_key : optional, str
        The path within the NeXuS file to the rotation angle data.

    Returns
    -------
    float
        The rotation angle at which the data in the NeXuS file was acquired at.
    """
    with h5py.File(str(file_path), 'r') as f:
        angle = f[angle_key][()]
    return angle


def _get_proj_max_dims(file_paths: List[Path], nxs_path: str
                       ) -> Tuple[int, int]:
    """Find the maximum value of the x dimension of all projections, and the
    maximum value of the y dimension of all projections.

    Parameters
    ----------
    file_paths : List[Path]
        A list of NeXuS files to combine.

    nxs_path : str
        The common path within the NeXuS files where the data to be combined is
        stored.

    Returns
    -------
    Tuple[int, int]
        The maximum x and y dimension values across all the projections.
    """
    x_dim = 0
    y_dim = 0

    for file_path in file_paths:
        with h5py.File(str(file_path), 'r') as proj:
            data = proj[nxs_path]
            if data.shape[0] > y_dim:
                y_dim = data.shape[0]
            if data.shape[1] > x_dim:
                x_dim = data.shape[1]

    return x_dim, y_dim


def _combine_proj_data(file_paths: List[Tuple[Path, float]], nxs_path:str,
                       x_dim:int, y_dim:int) -> Tuple[np.ndarray, np.ndarray]:
    """Combine projections from multiple NeXuS files into a single 3D numpy
    array.

    Parameters
    ----------
    file_paths : List[Tuple[Path, float]]
        A list of tuples containing:
            - The file path to a NeXuS file
            - The asscoaited rotation angle to the data in that NeXuS file

    nxs_path : str
        The common path within the NeXuS files where the data to be combined is
        stored.

    x_dim : int
        The size of the x dimension of the projections in the result.
    
    y_dim : int
        The size of the y dimension of the projections in the result.
    """
    no_of_angles = len(file_paths)
    angles = np.empty(no_of_angles)
    combined_data = np.empty((no_of_angles, y_dim, x_dim))
    pad_info = np.empty((no_of_angles, 4), dtype=int)

    # Iterate through all NeXuS files to combine their data into a single file
    # containing a stack of projections
    for idx, (file_path, _) in enumerate(file_paths):
        with h5py.File(str(file_path), 'r') as proj:
            data = proj[nxs_path][()]
        data = np.squeeze(data)

        # If a projection has smaller dimensions than the max, then
        # - pad the data using the minimum value
        # - place the smaller projection in the (approximate) centre of the max
        # dimensions
        if data.shape[0] != y_dim or data.shape[1] != x_dim:
            new_data = np.zeros((y_dim, x_dim)) + np.min(data)
            # calculate x and y shifts to center the smaller image within the
            # max dim values
            y_diff = y_dim - data.shape[0]
            y_shift = y_diff // 2
            x_diff = x_dim - data.shape[1]
            x_shift = x_diff // 2
            # y padding
            pad_info[idx, 0] = y_shift
            pad_info[idx, 1] = data.shape[0] + y_shift
            # x padding
            pad_info[idx, 2] = x_shift
            pad_info[idx, 3] = data.shape[1] + x_shift
            new_data[y_shift:data.shape[0] + y_shift,
                x_shift:data.shape[1] + x_shift] = data
        else:
            # y padding
            pad_info[idx, 0] = pad_info[idx, 1] = 0
            # x padding
            pad_info[idx, 2] = pad_info[idx, 3] = 0
            new_data = data

        combined_data[idx,:,:] = new_data

    return combined_data, pad_info


def _write_combined_proj_data(data:np.ndarray, angles:np.ndarray,
                              img_keys:np.ndarray, sample_desc:str,
                              file_path:str) -> None:
    """ Write the combined projection data to a NeXuS file.

    Parameters
    ----------
    data : np.ndarray
        The 3D numpy array containing the stack of projections.

    angles : np.ndarray
        The rotation angles of the stack of projections.

    img_keys : np.ndarray
        The image keys for the projections.

    sample_desc : str
        A description of the sample that the projection data was taken with.

    file_path : str
        The path to save the output NeXuS file to.
    """
    with h5py.File(file_path, 'w') as f:
        # some entries/metadata
        nxentry = f.create_group("tomo_entry")
        nxentry.attrs["NX_class"] = "NXentry"
        nxentry.attrs["definition"] = "NXtomo"

        nxinstrument = nxentry.create_group("instrument")
        nxinstrument.attrs["NX_class"] = "NXinstrument"

        # add image keys
        nxdetector = nxinstrument.create_group("detector")
        nxdetector.attrs["NX_class"] = "NXdetector"
        imgkey = nxdetector.create_dataset("image_key", data=img_keys)

        # add description of sample
        nxsample = nxentry.create_group("sample")
        nxsample.attrs["NX_class"] = "NXsample"
        nxsample.attrs["name"] = sample_desc

        # add 3D array containing combined data
        nxdata = nxentry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        phase = nxdata.create_dataset("data", data=data)
        phase.attrs["signal"] = [1,]

        # add rotation angles
        rotation = nxdata.create_dataset("rotation_angle", data=angles)
        rotation.attrs["units"] = "degrees"
        rotation.attrs["axis"] = [1,]


if __name__ == '__main__':
    main(sys.argv[1:])
