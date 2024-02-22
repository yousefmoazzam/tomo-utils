import sys
import os
import h5py
import click
from typing import List, NamedTuple, Tuple, Union
from pathlib import Path

import numpy as np


PROJ_IMG_KEY = 0
ELEMENT_PARENT_PATH = '/processed/auxiliary/0-XRF Elemental Maps from ROIs/'
XRF_DATA_KEY_TEMPLATE = lambda element: f"/processed/auxiliary/0-XRF Elemental Maps from ROIs/{element}/data"
XRF_ANGLE_PATH = "/entry/instrument/sample/sample_rot"
XRF_NXS_FILE = lambda i: f"i14-{i}-xsp3_addetector-xrf_windows-xsp3_addetector*.nxs"
DPC_DATA_KEY_TEMPLATE = lambda entry, dataset: f"/{entry}/{dataset}/data"
DPC_ANGLE_PATH = "/entry3/auxiliary/original_metadata/entry/instrument/sample/sample_rot"
DPC_NXS_FILE = lambda i: f"i14-{i}dpc.nxs"


class RegularScan(NamedTuple):
    path: Path


class HalfScans(NamedTuple):
    top_path: Path
    bottom_path: Path


ScanConfig = Union[RegularScan, HalfScans]


@click.command(
    help='A script to combine XRF or DPC data from multiple NeXuS files'
)
@click.option(
    "--data-type",
    required=True,
    type=click.Choice(["xrf", "dpc"], case_sensitive=False),
    help="Specify the type of input data to combine"
)
@click.option(
    '--nxs-dir',
    required=True,
    type=click.Path(),
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
    "--skip",
    required=False,
    multiple=True,
    type=int,
    default=(),
    help="Scan numbers to skip within the given range",
)
@click.option(
    "--stitch",
    required=False,
    multiple=True,
    nargs=2,
    type=int,
    help="Pair of scan numbers to stitch top/bottom halves",
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
@click.argument(
    "datasets",
    required=False,
    nargs=-1
)
def main(
    nxs_dir: Path,
    start_scan: int,
    end_scan: int,
    skip: Tuple[int],
    stitch: Tuple[Tuple[int, int]],
    sample_desc: str,
    out_dir: str,
    data_type: str,
    datasets: Tuple[str]
):
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

    skip : List[int]
        The scan numbers to skip.

    stitch : Tuple[Tuple[int, int]]
        A pair of scan numbers to stitch together (top and bottom halves).

    sample_desc : str
        A short description of the sample that will be place in the output NeXuS
        file metadata.

    out_dir : str
        The absolute path to the desired output directory.

    data_type : str
        The type of the input data to be combined.
    """
    if skip != ():
        scan_numbers = list(range(start_scan, end_scan+1))
        for scan_no in skip:
            if scan_no not in scan_numbers:
                warn_str = (
                    f"WARN: Scan number {scan_no} has been given to skip, "
                    f"but is not within the scan number range {start_scan} - "
                    f"{end_scan}, ignoring"
                )
                print(warn_str)

    # Define several things based on if the input data to combine is XRF or DPC:
    # - the string used to search for the NeXuS files to combine
    # - the path inside the NeXuS files to find the rotation angle information
    # - the function for getting the datasets within the NeXuS files
    if data_type.lower() == "xrf":
        _get_nxs_file = XRF_NXS_FILE
        angle_path = XRF_ANGLE_PATH
        _get_datasets = _get_xrf_datasets
    else:
        _get_nxs_file = DPC_NXS_FILE
        angle_path = DPC_ANGLE_PATH
        _get_datasets = _get_dpc_datasets

    # Check if the given directory exists
    nxs_dir = Path(nxs_dir)
    if not nxs_dir.exists():
        err_str = \
            f"The folder {nxs_dir} doesn't exist, please check your input"
        raise ValueError(err_str)

    # Get files in that dir whose scan number lies inside the given range
    nxs_file_paths = []
    for i in range(start_scan, end_scan+1):
        if i in skip:
            print(f"Skipping scan number {i}")
            continue
        search_term = _get_nxs_file(i)
        matches = list(nxs_dir.glob(search_term))
        if len(matches) == 1:
            nxs_file_paths.append(matches.pop())
        else:
            err_str = f"No NeXuS file with the scan number {i} in the given " \
                      f"range {start_scan}-{end_scan} was found in " \
                      f"{str(nxs_dir)}"
            raise ValueError(err_str)

    if len(datasets) > 0:
        # Verify that the given dataset(s) exist in the NeXuS files, by checking
        # the first NeXuS file in the scan number range for the dataset(s)
        with h5py.File(nxs_file_paths[0], "r") as f:
            all_datasets = []
            for group in datasets:
                try:
                    # Verify if the group containing the given dataset exists
                    group_lambda = lambda g: g if group in g else None
                    group_match = f["/"].visit(group_lambda)
                    assert group_match is not None
                except AssertionError:
                    err_str = f"The dataset \'{group}\' was not found in " \
                              f"the NeXuS file \'{nxs_file_paths[0]}\', " \
                              f"please double-check if it is correct (typos, " \
                              f"lowercase vs. uppercase letters, hyphens vs. " \
                              f"underscores, etc)?"
                    print(err_str)
                    sys.exit()

                try:
                    # Verify if the group contains a `/data` entry
                    dataset_lambda = lambda d: d if "data" in d else None
                    dataset_match = f[group_match].visit(dataset_lambda)
                    assert dataset_match is not None
                    if data_type == "xrf":
                        all_datasets.append(group)
                    else:
                        all_datasets.append((group_match.split("/")[0], group))
                except AssertionError:
                    err_str = f"The dataset \'{group}/data\' was not found " \
                              f"in the NeXuS file \'{nxs_file_paths[0]}\', " \
                              f"please double-check if {group} has a " \
                              f"\'/data\' entry?"
                    print(err_str)
                    sys.exit()
            info_str = ("All specified datasets have been verified to exist in "
                        "the NeXuS files")
            print(info_str)
    
    nxs_files_angles: List[Tuple[ScanConfig, float]] = []
    for (top, bottom) in stitch:
        top_filepath_idx = list(
            map(lambda path: str(top) in str(path), nxs_file_paths)
        ).index(True)
        top_filepath = nxs_file_paths.pop(top_filepath_idx)
        bottom_filepath_idx = list(
            map(lambda path: str(bottom) in str(path), nxs_file_paths)
        ).index(True)
        bottom_filepath = nxs_file_paths.pop(bottom_filepath_idx)
        angle = _get_rotation_angle(top_filepath, angle_path)
        nxs_files_angles.append((
            HalfScans(top_path=top_filepath, bottom_path=bottom_filepath),
            angle,
        ))

    # Check that all these files exist, and keep fetching the associated
    # rotation angle from them as long as they do exist
    for file_path in nxs_file_paths:
        if not file_path.exists():
            err_str = f"The file {file_path} doesn't exist."
            raise ValueError(err_str)
        else:
            angle = _get_rotation_angle(file_path, angle_path)
            nxs_files_angles.append(
                (RegularScan(file_path), angle)
            )
    
    print('All required files exist!')

    # Sort the nxs files in ascending order based on their associated rotation
    # angle
    nxs_files_angles.sort(key=lambda tup: tup[1])

    print('The NeXuS files have been sorted!')

    if len(datasets) == 0:
        # Find all different datasets in the NeXuS files (only need to search
        # one of the NeXuS files to find all datasets, since all NeXuS files
        # should contain the same datasets).
        with h5py.File(str(nxs_file_paths[0]), 'r') as f:
            all_datasets = _get_datasets(f)

    # Create the output dir if it doesn't already exist
    Path(out_dir).mkdir(exist_ok=True)

    img_keys = np.full(len(nxs_files_angles), PROJ_IMG_KEY)

    # Combine data for the given dataset from each separate NeXuS file
    for dataset in all_datasets:
        if data_type == "xrf":
            print(f"Combining element {dataset}...")
            filename = f"{dataset}.nxs"
            nxs_path = XRF_DATA_KEY_TEMPLATE(dataset)
        elif data_type == "dpc":
            print(f"Combining dataset {dataset[1]}")
            filename = f"{dataset[1]}.nxs"
            nxs_path = DPC_DATA_KEY_TEMPLATE(dataset[0], dataset[1])

        max_x_dim, max_y_dim = _get_proj_max_dims(
            [config for (config, _) in nxs_files_angles],
            nxs_path,
        )

        combined_data, pad_info = _combine_proj_data(
            [scan_config for (scan_config, _) in nxs_files_angles],
            nxs_path,
            max_x_dim,
            max_y_dim,
        )
        print('Saving file...')
        _write_combined_proj_data(
            combined_data,
            [angle for (path, angle) in nxs_files_angles],
            img_keys,
            sample_desc,
            Path(out_dir, filename),
            [scan_config for (scan_config, _) in nxs_files_angles],
        )
    print('Done!')


def _get_xrf_datasets(hdf5_file: h5py.File) -> List[str]:
    """
    Generate a list of strings describing the datasets in the given XRF NeXuS
    file.

    Example: suppose there is an XRF dataset with path
    "/processed/auxiliary/0-XRF Elemental Maps from ROIs/Ca-Ka/data" within the
    NeXuS files. The string associated with this dataset in the returned list
    would be "Ca-Ka".
    """
    return list(hdf5_file[ELEMENT_PARENT_PATH].keys())


def _get_dpc_datasets(hdf5_file: h5py.File) -> List[Tuple[str, str]]:
    """
    Generate a list of tuples describing the datasets in the given DPC NeXuS
    file.

    Example: suppose there is a DPC dataset with path "/entry1/Absorption/data"
    within the NeXuS files.

    The first entry in the associated tuple is the top-level group for this DPC
    dataset path (ie, "/entry1"). The second entry in the associated tuple is
    the second-level group for this DPC dataset path (ie, "Absorption").
    """
    entry_keys = list(hdf5_file["/"].keys())
    # The only dataset that is not needed from the DPC processed to be combined
    # is "/entry3/merlin_addetector/data", so it's safe to remove from the list
    # of keys
    entry_keys.remove("entry3")
    datasets = []
    dataset_lambda = lambda d: d if "data" in d else None
    for entry_key in entry_keys:
        dataset = hdf5_file[entry_key].visit(dataset_lambda)
        datasets.append((entry_key, dataset.split("/")[0]))

    return datasets


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


def _get_proj_max_dims(
    scan_configs: List[ScanConfig],
    nxs_path: str
) -> Tuple[int, int]:
    """Find the maximum value of the x dimension of all projections, and the
    maximum value of the y dimension of all projections.

    Parameters
    ----------
    scan_configs : List[ScanConfig]
        A list of `ScanConfig` objects that describe the associated NeXuS files
        to combine.

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

    for config in scan_configs:
        if isinstance(config, RegularScan):
            with h5py.File(str(config.path), 'r') as proj:
                data = proj[nxs_path]
                if data.shape[0] > y_dim:
                    y_dim = data.shape[0]
                if data.shape[1] > x_dim:
                    x_dim = data.shape[1]
        elif isinstance(config, HalfScans):
            stitched_y_len = 0
            with h5py.File(config.top_path, "r") as f:
                shape = f[nxs_path].shape
                stitched_x_len = shape[1]
                stitched_y_len += shape[0]
            with h5py.File(config.bottom_path, "r") as f:
                shape = f[nxs_path].shape
                stitched_y_len += shape[0]

            if stitched_y_len > y_dim:
                y_dim = stitched_y_len
            if stitched_x_len > x_dim:
                x_dim = stitched_x_len
        else:
            raise ValueError(
                f"Unsupported scan config type: {type(config)}"
            )

    return x_dim, y_dim


def _combine_proj_data(
    scan_configs: List[ScanConfig],
    nxs_path: str,
    x_dim: int,
    y_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine projections from multiple NeXuS files into a single 3D numpy
    array.

    Parameters
    ----------
    scan_configs : List[ScanConfig]
        A list of scan config objects that contain NeXuS file path info for the
        associated scan(s).

    nxs_path : str
        The common path within the NeXuS files where the data to be combined is
        stored.

    x_dim : int
        The size of the x dimension of the projections in the result.
    
    y_dim : int
        The size of the y dimension of the projections in the result.
    """
    no_of_angles = len(scan_configs)
    angles = np.empty(no_of_angles)
    combined_data = np.empty((no_of_angles, y_dim, x_dim))
    pad_info = np.empty((no_of_angles, 4), dtype=int)

    # Iterate through all NeXuS files to combine their data into a single file
    # containing a stack of projections
    for idx, scan_config in enumerate(scan_configs):
        if isinstance(scan_config, RegularScan):
            _get_regular_scan_data(
                idx,
                scan_config.path,
                nxs_path,
                combined_data,
                pad_info,
                x_dim,
                y_dim,
            )
        elif isinstance(scan_config, HalfScans):
            _get_half_scans_data(
                idx,
                scan_config.top_path,
                scan_config.bottom_path,
                nxs_path,
                combined_data,
                pad_info,
                x_dim,
                y_dim,
            )
        else:
            raise ValueError(
                f"Unsupported scan config type: {type(scan_config)}"
            )

    return combined_data, pad_info


def _get_regular_scan_data(
    idx: int,
    filepath: Path,
    data_path: str,
    combined_data: np.ndarray,
    pad_info: np.ndarray,
    x_dim: int,
    y_dim: int,
):
    """
    Gets projection data from the NeXuS file of a single "regular" scan
    ("regular" meaning that the scan completed during data acquisition with no
    issues).

    Note that this function modifies the `combined_data` and `pad_info` input.
    """
    with h5py.File(filepath, 'r') as proj:
        data = proj[data_path][()]
    data = np.squeeze(data)

    # If a projection has smaller dimensions than the max, then
    # - pad the data using the minimum value
    # - place the smaller projection in the (approximate) centre of the max
    # dimensions
    if data.shape[0] != y_dim or data.shape[1] != x_dim:
        new_data, pad_info[idx] = _pad_image(data, x_dim, y_dim)
    else:
        # y padding
        pad_info[idx, 0] = pad_info[idx, 1] = 0
        # x padding
        pad_info[idx, 2] = pad_info[idx, 3] = 0
        new_data = data

    combined_data[idx,:,:] = new_data


def _get_half_scans_data(
    idx: int,
    top_file_path: Path,
    bottom_file_path: Path,
    data_path: str,
    combined_data: np.ndarray,
    pad_info: np.ndarray,
    x_dim: int,
    y_dim: int,
) -> None:
    """
    Stitch top and bottom halves of partial scans.
    """
    X_VALUES_PATH = "/entry/It/SampleX_value_set"

    with h5py.File(top_file_path, "r") as f:
        top_shape = f[data_path].shape
        top_x_values = f[X_VALUES_PATH][:]

    with h5py.File(bottom_file_path, "r") as f:
        bottom_shape = f[data_path].shape
        bottom_x_values = f[X_VALUES_PATH][:]

    stitched_scan_height: int = top_shape[0] + bottom_shape[0]
    stitched_scan_width: int = max(top_shape[1], bottom_shape[1])
    stitched_scan = np.empty((stitched_scan_height, stitched_scan_width))

    with h5py.File(top_file_path, "r") as f:
        dataset: h5py.Dataset = f[data_path]
        top_data: np.ndarray = dataset[:]

    stitched_scan[0:top_shape[0], :] = top_data

    with h5py.File(bottom_file_path, "r") as f:
        dataset: h5py.Dataset = f[data_path]
        bottom_data: np.ndarray = dataset[:]

    diff = np.abs(top_x_values - bottom_x_values[0])
    min_val_idx = np.argmin(diff)

    bottom_half_container: np.ndarray = np.zeros((bottom_shape[0], top_shape[1]))
    bottom_half_container[:, :bottom_data.shape[1]] = bottom_data
    stitched_scan[top_shape[0]:, :] = np.roll(
        bottom_half_container,
        shift=min_val_idx,
        axis=1,
    )

    if stitched_scan.shape[0] != y_dim or stitched_scan.shape[1] != x_dim:
        stitched_scan, pad_info[idx] = _pad_image(stitched_scan, x_dim, y_dim)
    else:
        # y padding
        pad_info[idx, 0] = pad_info[idx, 1] = 0
        # x padding
        pad_info[idx, 2] = pad_info[idx, 3] = 0

    combined_data[idx,:,:] = stitched_scan


def _pad_image(
    image: np.ndarray,
    x: int,
    y: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to match the given dimensions
    """
    min_value = np.min(image)
    if np.isnan(min_value):
        min_value = 0
    padded_image = np.zeros((y, x)) + min_value
    pad_info = np.zeros(4)

    # calculate x and y shifts to center the smaller image within the
    # max dim values
    y_diff = y - image.shape[0]
    y_shift = y_diff // 2
    x_diff = x - image.shape[1]
    x_shift = x_diff // 2

    # y padding
    pad_info[0] = y_shift
    pad_info[1] = image.shape[0] + y_shift

    # x padding
    pad_info[2] = x_shift
    pad_info[3] = image.shape[1] + x_shift

    padded_image[
        y_shift:image.shape[0] + y_shift,
        x_shift:image.shape[1] + x_shift
    ] = image

    return padded_image, pad_info


def _write_combined_proj_data(
    data:np.ndarray,
    angles:np.ndarray,
    img_keys:np.ndarray,
    sample_desc:str,
    file_path:str,
    scan_configs: List[ScanConfig],
) -> None:
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

    scan_configs : List[ScanConfig]
        A list of scan config objects that contain NeXuS file path info for the
        associated scan(s).
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

        # add filepaths where combined scans came from
        hdf5_str_type = h5py.special_dtype(vlen=str)
        filepaths = np.array([
            str(config.path)
            if isinstance(config, RegularScan)
            else np.array([
                str(config.top_path), str(config.bottom_path)
                ], dtype=hdf5_str_type)
            for config in scan_configs
        ], dtype=hdf5_str_type)
        nxdata.attrs["origin"] = filepaths


if __name__ == '__main__':
    main()
