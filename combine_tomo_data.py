import sys
import os
import h5py
import click
from typing import List, Tuple
from pathlib import Path

import numpy as np


# Assuming XRF data for now, so data key is templated due to the different
# elemental map data needing to be combined
DATA_KEY_TEMPLATE = lambda element: f"/processed/auxiliary/0-XRF Elemental Maps from ROIs/{element}/data"
PROJ_IMG_KEY = 0
ELEMENT_PARENT_PATH = '/processed/auxiliary/0-XRF Elemental Maps from ROIs/'


@click.command(help='A script to combine XRF data from multiple NeXuS files')
@click.option(
    '--nxs-dir',
    required=True,
    type=click.Path(path_type=Path),
    help='Folder containing the NeXuS files to combine'
)
@click.option(
    '--start-scan',
    required=True,
    type=int,
    help='Scan number marking the start of the range of NeXuS files to combine'
)
@click.option(
    '--end-scan',
    required=True,
    type=int,
    help='Scan number marking the end of the range of NeXuS files to combine'
)
@click.option(
    '--out-dir',
    required=True,
    type=str,
    help="Desired folder to contain the output (will be created if it doesn't exist)"
)
@click.option(
    '--sample-desc',
    required=True,
    type=str,
    help='Short description of sample to be put into output NeXuS file metadata'
)
def main(nxs_dir: Path, start_scan: int, end_scan: int, sample_desc: str,
         out_dir: str):
    """
    Parameters
    ----------
    nxs_dir : Path
        The absolute path to the directory containing the different NeXuS files
        to combine.

    start_scan_no : int
        The scan number marking the start of the range of NeXuS files to
        combine.

    end_scan_no : int
        The scan number marking the end of the range of NeXuS files to combine.

    sample_desc : str
        A short description of the sample that will be place in the output NeXuS
        file metadata.

    out_dir : str
        The absolute path to the desired output directory.
    """
    # Check if the given directory exists
    if not nxs_dir.exists():
        err_str = \
            f"The folder {nxs_dir} doesn't exist, please check your input"
        raise ValueError(err_str)

    # Get files in that dir whose scan number lies inside the given range
    nxs_file_paths = []
    for i in range(start_scan, end_scan+1):
        glob_str = f"i14-{i}-xsp3_addetector-xrf_windows-xsp3_addetector*.nxs"
        matches = list(nxs_dir.glob(glob_str))
        if len(matches) == 1:
            nxs_file_paths.append(matches.pop())
        else:
            err_str = f"No NeXuS file with the scan number {i} in the given " \
                      f"range {start_scan}-{end_scan} was found in " \
                      f"{str(nxs_dir)}"
            raise ValueError(err_str)
    
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

    # Find all different element maps in the NeXuS files (only need to search
    # one of the NeXuS files to find all element maps, since all NeXuS files
    # should contain the same element maps).
    with h5py.File(str(nxs_file_paths[0]), 'r') as f:
        all_element_maps = list(f[ELEMENT_PARENT_PATH].keys())

    # Create the output dir if it doesn't already exist
    Path(out_dir).mkdir(exist_ok=True)

    img_keys = np.full(len(nxs_files_angles), PROJ_IMG_KEY)

    # Combine data for all element maps, each one being in a separate NeXuS file
    for element in all_element_maps:
        print(f"Combining {element} data...")
        filename = f"{element}.nxs"

        max_x_dim, max_y_dim = \
            _get_proj_max_dims(nxs_file_paths, DATA_KEY_TEMPLATE(element))

        combined_data, pad_info = \
            _combine_proj_data(nxs_files_angles, DATA_KEY_TEMPLATE(element),
                               max_x_dim, max_y_dim)
        print('Saving file...')
        _write_combined_proj_data(combined_data,
                                  [angle for (path, angle) in nxs_files_angles],
                                  img_keys, sample_desc, Path(out_dir, filename))
    print('Done!')


def _get_rotation_angle(file_path: Path,
                        angle_key: str='/entry/instrument/sample/sample_rot'
                        ) -> float:
    """Get the rotation angle associated with the given NeXuS file.

    Parameters
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
        nxentry = f.create_group('entry')
        nxentry.attrs["NX_class"] = "NXentry"
        nxtomoentry = nxentry.create_group("tomo_entry")
        nxtomoentry.attrs["NX_class"] = "NXsubentry"

        nxinstrument = nxtomoentry.create_group("instrument")
        nxinstrument.attrs["NX_class"] = "NXinstrument"

        # add image keys
        nxdetector = nxinstrument.create_group("detector")
        nxdetector.attrs["NX_class"] = "NXdetector"
        imgkey = nxdetector.create_dataset("image_key", data=img_keys)

        # add description of sample
        nxsample = nxtomoentry.create_group("sample")
        nxsample.attrs["NX_class"] = "NXsample"
        nxsample.attrs["name"] = sample_desc

        # add 3D array containing combined data
        nxdata = nxtomoentry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        phase = nxdata.create_dataset("data", data=data)
        phase.attrs["signal"] = [1,]

        # add rotation angles
        rotation = nxdata.create_dataset("rotation_angle", data=angles)
        rotation.attrs["units"] = "degrees"
        rotation.attrs["axis"] = [1,]


if __name__ == '__main__':
    main()
