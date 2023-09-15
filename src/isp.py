"""Image Processing Pipeline/"""
import argparse as ap
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import interp2d
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import rescale


def read_im(im_path: Path) -> np.ndarray:
    """Reads image file.
  
    Args:
        im_path: Path to image file.

    Returns:
        Image read into double precision numpy array. 
    """
    im = imread(im_path)
    print(f"This image has {im[0,0].nbytes * 8} bits per pixel.")
    im = im.astype(np.double)
    print(f"This image now has {im[0,0].nbytes * 8} bits per pixel.")

    return im


def write_im(out_path: Path, im: np.ndarray, compression_quality: int):
    """"Writes image to file.
    
    Args:
        out_path: Path to write image to.
        im: Array containing image data to write.
        compression_quality: Parameter that defines compression quality when saving as JPEG.
    """
    # Clip image to [0, 1] range.
    im = np.clip(im, 0.0, 1.0)
    im = (im * 255.0).astype(np.uint8)

    # Checks to see whether image is grayscale or RGB.
    if len(im.squeeze().shape) == 3:
        im = Image.fromarray(im).convert("RGB")
    else:
        im = Image.fromarray(im).convert("L")

    if out_path.suffix == ".png":
        im.save(out_path)
    elif out_path.suffix == ".jpg":
        im.save(out_path.parent /
                f"{out_path.stem}_{compression_quality}{out_path.suffix}",
                quality=compression_quality)
    else:
        raise RuntimeError("Incompatible image format!")


def linearize(image: np.ndarray) -> np.ndarray:
    """Linearizes image based on dcraw black and white values.
  
    Args:
        image: Non-linear image.

    Returns:
        Linearized image.
    """

    black = 150
    white = 4095
    # Normalize so that black is mapped to 0 and white is mapped to 1.0.
    linear_image = (image - black) / (white - black)
    # Clip image to [0, 1].
    linear_image = np.clip(linear_image, 0.0, 1.0)

    return linear_image


def get_bayer_data(
        image: np.ndarray,
        bayer_pattern: str = "rggb") -> (np.ndarray, np.ndarray, np.ndarray):
    """Function to get useful data according to specific Bayer pattern.

    Provides binary masks and grid coordinates that are used for white balancing and demosaicing.

    Args:
        image: Linear, mosaiced image.
        bayer_pattern: One of ["grbg", "rggb", "bggr", "gbrg"] that represents  Bayer pattern.
    
    Returns:
        3 binary masks and set of grid coordinates corresponding to Bayer pattern. 
    """

    H, W = image.shape
    # Coordinate ranges based on 2x2 Bayer grid unit.
    top_left = (np.arange(0, H, 2), np.arange(0, W, 2))
    top_right = (np.arange(0, H, 2), np.arange(1, W, 2))
    bottom_left = (np.arange(1, H, 2), np.arange(0, W, 2))
    bottom_right = (np.arange(1, H, 2), np.arange(1, W, 2))

    # Full set of coordinates based on Bayer pattern.
    top_left_grid = np.meshgrid(*top_left)
    top_right_grid = np.meshgrid(*top_right)
    bottom_left_grid = np.meshgrid(*bottom_left)
    bottom_right_grid = np.meshgrid(*bottom_right)

    if bayer_pattern == "grbg":
        green_y1, green_x1 = top_left_grid
        red_y, red_x = top_right_grid
        blue_y, blue_x = bottom_left_grid
        green_y2, green_x2 = bottom_right_grid
    elif bayer_pattern == "rggb":
        red_y, red_x = top_left_grid
        green_y1, green_x1 = top_right_grid
        green_y2, green_x2 = bottom_left_grid
        blue_y, blue_x = bottom_right_grid
    elif bayer_pattern == "bggr":
        blue_y, blue_x = top_left_grid
        green_y1, green_x1 = top_right_grid
        green_y2, green_x2 = bottom_left_grid
        red_y, red_x = bottom_right_grid
    elif bayer_pattern == "gbrg":
        green_y1, green_x1 = top_left_grid
        blue_y, blue_x = top_right_grid
        red_y, red_x = bottom_left_grid
        green_y2, green_x2 = bottom_right_grid
    else:
        raise RuntimeError("Invalid Bayer pattern!")

    red_mask = np.zeros_like(image).astype(bool)
    blue_mask = np.zeros_like(image).astype(bool)
    green_mask = np.zeros_like(image).astype(bool)

    red_mask[red_y.T, red_x.T] = 1
    blue_mask[blue_y.T, blue_x.T] = 1
    green_mask[green_y1.T, green_x1.T] = 1
    green_mask[green_y2.T, green_x2.T] = 1

    red_coords = (np.arange(red_y.min(), red_y.max(),
                            2), np.arange(red_x.min(), red_x.max(), 2))
    blue_coords = (np.arange(blue_y.min(), blue_y.max(),
                             2), np.arange(blue_x.min(), blue_x.max(), 2))
    green_coords1 = (np.arange(green_y1.min(), green_y1.max(),
                               2), np.arange(green_x1.min(), green_x1.max(),
                                             2))
    green_coords2 = (np.arange(green_y2.min(), green_y2.max(),
                               2), np.arange(green_x2.min(), green_x2.max(),
                                             2))

    return (red_coords, red_mask), (green_coords1, green_coords2,
                                    green_mask), (blue_coords, blue_mask)


def white_balance(linear_image: np.ndarray,
                  method: str = "white",
                  bayer_pattern: str = "rggb") -> np.ndarray:
    """White balances image with specified AWB method and Bayer pattern.
    
    Args:
      image: Linear image to be white balanced.
      method: One of ["white", "grey", "preset"] indicating which auto white balance method to use
      bayer_pattern: One of ["grbg", "rggb", "bggr", "gbrg"] that represents  Bayer pattern.
    
    Returns:
      White balanced image.
    """
    if not method == "manual":
        red_data, green_data, blue_data = get_bayer_data(
            linear_image, bayer_pattern=bayer_pattern)
        red_mask = red_data[-1]
        green_mask = green_data[-1]
        blue_mask = blue_data[-1]

    if method == "white":
        r_max = linear_image[red_mask].max()
        g_max = linear_image[green_mask].max()
        b_max = linear_image[blue_mask].max()

        linear_image[red_mask] = linear_image[red_mask] * g_max / r_max
        linear_image[blue_mask] = linear_image[blue_mask] * b_max / r_max
    elif method == "grey":
        r_avg = linear_image[red_mask].mean()
        g_avg = linear_image[green_mask].mean()
        b_avg = linear_image[blue_mask].mean()

        linear_image[red_mask] = linear_image[red_mask] * g_avg / r_avg
        linear_image[blue_mask] = linear_image[blue_mask] * g_avg / b_avg
    elif method == "preset":
        linear_image[red_mask] = linear_image[red_mask] * 2.394531
        linear_image[blue_mask] = linear_image[blue_mask] * 1.597656
    elif method == "manual":
        if not linear_image.shape[-1] == 3:
            raise RuntimeError(
                "For manual white balancing, please pass in an RGB image.")

        # Create figure for manual selection.
        _, ax = plt.subplots()
        # Downsample image for display purposes.
        downscale = 8
        display_image = rescale(linear_image, 1 / downscale)
        ax.imshow(display_image, cmap="gray")
        top_left, bottom_right = plt.ginput(n=2, timeout=0)
        plt.close("all")

        # Draw Rectangle on Image and Show
        _, ax = plt.subplots()
        rect = patches.Rectangle(top_left,
                                 bottom_right[0] - top_left[0],
                                 top_left[1] - bottom_right[1],
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.imshow(display_image, cmap="gray")
        plt.show()

        # Extract white patch from image.
        start_y = int(top_left[1]) * downscale
        end_y = int(bottom_right[1]) * downscale
        start_x = int(top_left[0]) * downscale
        end_x = int(bottom_right[0]) * downscale
        white_patch = linear_image[start_y:end_y, start_x:end_x]

        # Normalize and clip image based on white patch.
        linear_image /= white_patch.max(axis=(0, 1))
        linear_image = np.clip(linear_image, 0.0, 1.0)

        # Draw patch on image.
        pad = 8
        linear_image[start_y:end_y + pad, start_x:start_x + pad] = [1, 0, 0]
        linear_image[start_y:start_y + pad, start_x:end_x + pad] = [1, 0, 0]
        linear_image[start_y:end_y + pad, end_x:end_x + pad] = [1, 0, 0]
        linear_image[end_y:end_y + pad, start_x:end_x + pad] = [1, 0, 0]
    else:
        raise RuntimeError(
            "Please select one of [white, grey, preset] for white balancing!")
    return linear_image


def demosaic(white_balanced_image: np.ndarray,
             bayer_pattern: str) -> np.ndarray:
    """Demosaics white balanced, mosaiced image.

    Args:
        white_balanced_image: White-balanced, mosaiced image.
        bayer_pattern: One of ["grbg", "rggb", "bggr", "gbrg"] that represents  Bayer pattern.
    
    Returns:
        Demosaiced image.
    """

    H, W = white_balanced_image.shape
    red_data, green_data, blue_data = get_bayer_data(
        white_balanced_image, bayer_pattern=bayer_pattern)

    # Red channel.
    (red_y, red_x), red_mask = red_data
    red_coords_y, red_coords_x = np.meshgrid(red_y, red_x)
    red_data = white_balanced_image[red_coords_y.T, red_coords_x.T]
    red_interpolator = interp2d(red_x, red_y, red_data, kind="linear")
    red_channel = red_interpolator(np.arange(W), np.arange(H))
    red_channel[red_mask] = white_balanced_image[red_mask]

    # Blue channel.
    (blue_y, blue_x), blue_mask = blue_data
    blue_coords_y, blue_coords_x = np.meshgrid(blue_y, blue_x)
    blue_data = white_balanced_image[blue_coords_y.T, blue_coords_x.T]
    blue_interpolator = interp2d(blue_x, blue_y, blue_data, kind="linear")
    blue_channel = blue_interpolator(np.arange(W), np.arange(H))
    blue_channel[blue_mask] = white_balanced_image[blue_mask]

    # Green channel.
    (green_y1, green_x1), (green_y2, green_x2), green_mask = green_data
    green_coords_y, green_coords_x = np.meshgrid(green_y1, green_x1)
    green_data = white_balanced_image[green_coords_y.T, green_coords_x.T]
    green_interpolator1 = interp2d(green_x1,
                                   green_y1,
                                   green_data,
                                   kind="linear")
    green_channel1 = green_interpolator1(np.arange(W), np.arange(H))

    green_coords_y, green_coords_x = np.meshgrid(green_y2, green_x2)
    green_data = white_balanced_image[green_coords_y.T, green_coords_x.T]
    green_interpolator2 = interp2d(green_x2,
                                   green_y2,
                                   green_data,
                                   kind="linear")
    green_channel2 = green_interpolator2(np.arange(W), np.arange(H))

    green_channel = (green_channel1 + green_channel2) / 2
    green_channel[green_mask] = white_balanced_image[green_mask]

    demosaiced_im = np.dstack([red_channel, green_channel, blue_channel])
    return demosaiced_im


def color_space_correction(demosaiced_image: np.ndarray) -> np.ndarray:
    """Performs color space correction on demosaiced image.

    Args:
        demosaiced_image: Demosaiced image to perform color space correction on.
    
    Returns:
        Color space corrected image.
    """

    M_srgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
    # For Nikon D3400
    M_xyz_to_cam = np.array([[6988, -1384, -714], [-5631, 13410, 2447],
                             [-1485, 2204, 7318]]) / 10000.0
    M_xyz_to_cam = M_xyz_to_cam

    M_srgb_to_cam = M_xyz_to_cam @ M_srgb_to_xyz
    M_srgb_to_cam = M_srgb_to_cam / M_srgb_to_cam.sum(axis=1).reshape(-1, 1)
    M_srgb_to_cam_inv = np.linalg.inv(M_srgb_to_cam)

    color_space_corrected_img = (
        M_srgb_to_cam_inv @ (demosaiced_image.reshape(-1, 3)).T).T.reshape(
            demosaiced_image.shape)

    return color_space_corrected_img


def brighten_image(color_space_corrected_image: np.ndarray,
                   brightness_percentage: float = 0.25) -> np.ndarray:
    """Brightens image based on brightness percentage.
    
    Linearly scales image values such that grayscale mean after brightening = brightness_percentage.

    Args:
        color_space_corrected_image: Color corrected image.
        brightness_percentage: "Percentage" by which to brighten image.

    Returns:
        Brightened image.
    """
    gray = rgb2gray(color_space_corrected_image)
    pre_brightening_gray_mean = gray.mean()

    post_brightening_gray_mean = brightness_percentage

    scale = post_brightening_gray_mean / pre_brightening_gray_mean
    brightened_image = color_space_corrected_image * scale
    brightened_image = np.clip(brightened_image, 0.0, 1.0)
    return brightened_image


def gamma_encoding(brightened_image: np.ndarray) -> np.ndarray:
    """Gamma encodes image.
    
    Applies following non-linearity to image:

        12.92 * C_linear if C_linear <= 0.0031308
        (1 + 0.055) * C_linear**(1/2.4) - 0.055 if C_linear > 0.0031308

        where C_linear = {R, G, B} of brightened_image

    Args:
        brightened_image: Brightened image.

    Returns:
        Gamma encoded image.

    """
    # Indices where C_linear <= 0.0031308
    cond1_coords = np.where(brightened_image <= 0.0031308)
    # Indices where C_linear > 0.0031308
    cond2_coords = np.where(brightened_image > 0.0031308)

    brightened_image[cond1_coords] *= 12.92
    brightened_image[cond2_coords] = (
        (1 + 0.055) *
        np.power(brightened_image[cond2_coords], 1 / 2.4)) - 0.055

    return brightened_image


def tone_reproduction(color_space_corrected_image: np.ndarray,
                      brightness_percentage: float = 0.25) -> np.ndarray:
    """Full tone reproduction function.

    Connects brightening and gamma encoding in one function.

    Args:
        color_space_corrected_image: Color corrected image.
        brightness_percentage: "Percentage" to brighten image by.
    """
    brightened_image = brighten_image(color_space_corrected_image,
                                      brightness_percentage)
    gamma_encoded_image = gamma_encoding(brightened_image)

    return gamma_encoded_image


def main(args):
    """Main driver for image processing pipeline."""

    # Parse command line arguments.
    src_dir = Path(args.src_dir)
    bayer_pattern = str(args.bayer_pattern)
    white_balance_mode = str(args.white_balance)
    brightness_percentage = float(args.brightness_percentage) / 100.0
    # Create output directory based on settings.
    out_dir = Path(args.out_dir) / white_balance_mode / bayer_pattern
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)

    output_img_format = str(args.output_img_format)
    compression_quality = int(args.compression_quality)

    im = read_im(src_dir / "campus.tiff")
    print("Linearizing...")
    linear_im = linearize(im)

    print("White Balancing...")
    if not white_balance_mode == "manual":
        white_balanced_im = white_balance(linear_im,
                                          bayer_pattern=bayer_pattern,
                                          method=white_balance_mode)
        write_im(out_dir / f"1_white_balanced.{output_img_format}",
                 white_balanced_im, compression_quality)
    else:
        white_balanced_im = linear_im

    print("Demosaicing...")
    demosaiced_image = demosaic(white_balanced_im, bayer_pattern=bayer_pattern)
    write_im(out_dir / f"2_demosaiced.{output_img_format}", demosaiced_image,
             compression_quality)

    if white_balance_mode == "manual":
        demosaiced_image = white_balance(demosaiced_image,
                                         bayer_pattern=bayer_pattern,
                                         method=white_balance_mode)
        write_im(out_dir / f"1_white_balanced.{output_img_format}",
                 white_balanced_im, compression_quality)

    print("Correcting Color Space...")
    color_space_corrected_img = color_space_correction(demosaiced_image)
    write_im(out_dir / f"3_color_corrected.{output_img_format}",
             color_space_corrected_img, compression_quality)

    print("Peforming Tone Reproduction...")
    gamma_encoded_image = tone_reproduction(color_space_corrected_img,
                                            brightness_percentage)
    write_im(
        out_dir /
        f"4_gamma_encoded_{int(brightness_percentage*100)}.{output_img_format}",
        gamma_encoded_image, compression_quality)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--src_dir",
                        type=str,
                        default="../data",
                        required=True)
    parser.add_argument("--bayer_pattern",
                        type=str,
                        default="rggb",
                        required=True)
    parser.add_argument("--white_balance",
                        type=str,
                        default="grey",
                        required=True)
    parser.add_argument("--brightness_percentage",
                        type=float,
                        default=20,
                        required=True)
    parser.add_argument("--output_img_format",
                        type=str,
                        default="png",
                        required=True)
    parser.add_argument("--compression_quality",
                        type=int,
                        default=95,
                        required=False)
    parser.add_argument("--out_dir", type=str, default="./data", required=True)
    args = parser.parse_args()

    main(args)