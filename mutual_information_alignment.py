import sys
import click
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from skimage import transform as tf

# Import of `estimate_shift2D_mutual` must come before the `sys.path.insert()`
# for the hyperspy repo, otherwise there are import issues for the
# `estimate_shift2D_mutual` function
from i14_utility.xanes.mutual import estimate_shift2D_mutual
# TODO: This is currently a workaround to not having a system-wide conda env
# with a compatible version of Hyperspy installed in it, and it's dependent on a
# local clone of the git repo which can be changed at any time, so it's not a
# robust solution...
sys.path.insert(0, '/dls_sw/i14/scripts/i14_python/tomo_utility/hyperspy')
import hyperspy.api as hs


@click.command(
    help='Apply mutual information alignment on a stack of tomography projections'
)
@click.argument(
    'in_file', type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    'out_file', type=click.Path(exists=False, dir_okay=False, path_type=Path)
)
@click.option(
    '--roi',
    nargs=4,
    type=click.Tuple([int, int, int, int]),
    default=None,
    help='Specify ROI in which the sample stays within across all projections'
)
@click.option(
    '--reference',
    type=click.Choice(['current', 'cascade', 'stat']),
    default='current',
    help="If 'current' (default) the image at the current " \
        "coordinates is taken as reference. If 'cascade' each image " \
        "is aligned with the previous one. If 'stat' the translation " \
        "of every image with all the rest is estimated and by " \
        "performing statistical analysis on the result the " \
        "translation is estimated."
)
@click.option(
    '--save-shifts',
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help='File to save the calculated shifts in'
)
def main(in_file: Path, out_file: Path, roi: Tuple[int, int, int, int],
         reference: str, save_shifts: Path):
    """
    Parameters
    ----------
    roi : Tuple[int, int, int, int]
        Define the region of interest.

    reference : str, {'current', 'cascade' ,'stat'}
        If 'current' (default) the image at the current coordinates is taken as
        reference. If 'cascade' each image is aligned with the previous one. If
        'stat' the translation of every image with all the rest is estimated and
        by performing statistical analysis on the result the translation is
        estimated.
    """
    # Load the combined data
    print('Loading data...')
    with h5py.File(in_file, 'r') as f:
        data = f['/entry/tomo_entry/data/data'][:, :, :]
        rot_angles = f['/entry/tomo_entry/data/rotation_angle'][:]
        im_key = f['/entry/tomo_entry/instrument/detector/image_key'][:]

    # Create Signal2D objct to hold the data
    signal = hs.signals.Signal2D(data)

    # Get the shifts that are needed to align the data
    print('Calculating shifts...')
    shifts, max_vals = \
        estimate_shift2D_mutual(signal, reference=reference, roi=roi)

    # Save shifts somewhere
    if save_shifts is not None:
        np.save(save_shifts, shifts)

    # Perform alignment
    print('Performing alignment...')
    shifted_data = np.zeros(shape=data.shape)
    for i in range(data.shape[0]):
        x_shift = shifts[i][1]
        y_shift = shifts[i][0]
        transformation = \
            tf.SimilarityTransform(translation=(x_shift, y_shift))
        transformed_image = tf.warp(data[i], transformation, order=0)
        shifted_data[i,:,:] = transformed_image

    # Write the aligned projections to an hdf file
    print('Saving output...')
    with h5py.File(out_file, 'a') as f_out:
        f_out.create_dataset('/entry/tomo_entry/data/data', data=shifted_data)
        f_out.create_dataset('/entry/tomo_entry/data/rotation_angle',
                             data=rot_angles)
        f_out.create_dataset('/entry/tomo_entry/instrument/detector/image_key',
                             data=im_key)


if __name__ == '__main__':
    main()
