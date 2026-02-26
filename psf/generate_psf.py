import numpy as np
import tifffile
from pathlib import Path

def generate_optical_psf(dxy, dz, SizeXY, SizeZ, wavelength, NA, RI):
    """Generate optical PSF using Born & Wolf theory.

    Args:
        dxy, dz: Pixel size in nm
        SizeXY, SizeZ: PSF dimensions in pixels
        wavelength: Emission wavelength in nm
        NA: Numerical aperture
        RI: Refractive index

    Returns:
        PSF array (Y, X, Z)
    """
    dk = 2 * np.pi / dxy / SizeXY
    kx = np.arange(-(SizeXY-1)//2, (SizeXY-1)//2 + 1) * dk
    kx, ky = np.meshgrid(kx, kx)
    kr_sq = kx**2 + ky**2
    z = np.arange(-(SizeZ-1)//2, (SizeZ-1)//2 + 1) * dz

    PupilMask = (kr_sq <= (2*np.pi/wavelength*NA)**2)
    kz_temp = (2*np.pi/wavelength*RI)**2 - kr_sq
    kz_temp = np.where(kz_temp >= 0, kz_temp, 0)
    kz = np.sqrt(kz_temp) * PupilMask

    PSF = np.zeros((SizeXY, SizeXY, SizeZ), dtype=np.float64)

    for ii in range(SizeZ):
        phase_term = PupilMask * np.exp(1j * kz * z[ii])
        fft_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phase_term)))
        PSF[:, :, ii] = np.abs(fft_result)**2

    return PSF

def generate_gaussian_psf(size, sigma):
    """Generate normalized Gaussian PSF.

    Args:
        size: Kernel size in pixels
        sigma: Standard deviation in pixels

    Returns:
        PSF array (size, size, 1)
    """
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)

    gaussian_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
    gaussian_3d = gaussian_2d[:, :, np.newaxis]

    return gaussian_3d

def save_psf_tiff(psf, output_path):
    """Save PSF as 16-bit TIFF.

    Args:
        psf: PSF array
        output_path: Output file path

    Returns:
        Path to saved file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    psf_normalized = psf / np.max(psf) * (2**15)
    psf_uint16 = psf_normalized.astype(np.uint16)

    if psf.shape[2] == 1:
        tifffile.imwrite(str(output_path), psf_uint16[:, :, 0])
    else:
        psf_transposed = np.transpose(psf_uint16, (2, 0, 1))
        tifffile.imwrite(str(output_path), psf_transposed)

    return str(output_path)

def create_optical_psf_file(dxy=92.6, dz=92.6, SizeXY=257, SizeZ=1, wavelength=525, NA=1.1, RI=1.3):
    """Generate and save optical PSF with auto-generated filename."""
    psf = generate_optical_psf(dxy, dz, SizeXY, SizeZ, wavelength, NA, RI)

    filename = f"PSF_optical_NA{NA}_lambda{wavelength}_size{SizeXY}"
    if SizeZ > 1:
        filename += f"_Z{SizeZ}"
    filename += ".tif"

    base_dir = Path(__file__).parent
    output_path = base_dir / "PSFoutput" / "optical" / filename
    saved_path = save_psf_tiff(psf, output_path)

    print(f"Generated optical PSF: {filename}")

    return saved_path

def create_gaussian_psf_file(size=31, sigma=3.0):
    """Generate and save Gaussian PSF with auto-generated filename."""
    psf = generate_gaussian_psf(size, sigma)

    filename = f"PSF_gaussian_sigma{sigma}_size{size}.tif"

    base_dir = Path(__file__).parent
    output_path = base_dir / "PSFoutput" / "gaussian" / filename
    saved_path = save_psf_tiff(psf, output_path)

    print(f"Generated Gaussian PSF: {filename} (sigma={sigma} pixels)")

    return saved_path

if __name__ == "__main__":
    create_optical_psf_file(dxy=31.3, dz=31.3, SizeXY=79, SizeZ=8, wavelength=525, NA=5.0, RI=1.3)
    # create_gaussian_psf_file(size=257, sigma=4.0)