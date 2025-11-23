import os
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.ndimage
import logging

import desispec.io
from desispec import coaddition

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset # add as in docs

import einops
from torch.utils.data import Dataset


def find_matching_indices(targets, reference_ids):
    """Return indices of `targets` in `reference_ids`."""
    id_to_index = {tid: i for i, tid in enumerate(reference_ids)}
    return np.array([id_to_index[tid] for tid in targets])

class EuclidDESIDataset(Dataset):
    def __init__(self, 
                 metadata_path, 
                 vis_folder, 
                 nisp_folders, 
                 spectra_folder,
                 healpix_nside=64,
                 transform={}, 
                 stochastic=True,
                 spiral=False, 
                 modality_registry=None
        ):
        """
        Args:
            metadata_path (str): Path to base_EuclidQ1_DESIDR1.fits file.
            vis_folder (str): Path to folder with VIS images.
            nisp_folder (str): Path to folder with NISP images (H, J, Y).
            healpix_nside (int): NSIDE for HEALPix structure (informational only).
        """
        self.meta = Table.read(metadata_path)
        self.vis_folder = vis_folder
        self.nisp_folder = nisp_folders
        self.spectra_folder = spectra_folder
        
        self.healpix_nside = healpix_nside
        
        self.transform = transform
        self.stochastic = stochastic
        self.spiral = spiral
        self.modality_registry = modality_registry
        
        self.dataset_len = len(self.meta) 
        self.stochastic = False

    def __len__(self):
        return len(self.meta)
        
    @staticmethod
    def _spiral(n):
        """
        generate a spiral index array of side length 'n'
        there must be a better way to do this: any suggestions?
        """
        a = np.arange(n * n)
        b = a.reshape((n, n))
        m = None
        for i in range(n, 0, -2):
            m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
            b = b[1:-1, 1:-1]
        a[list(m[1:])] = list(a)
        a = abs(a - n * n + 1)
        return a.reshape((n, n))
    
    def spiralise(self, galaxy):
        """
        Change ViT patch ordering to a 'spiral order'. See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        Alternate function available here:
        https://www.procook.co.uk/product/procook-spiralizer-black-and-stainless-steel
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        spiraled = [ii for _, ii in sorted(zip(indices, galaxy))]
        return (
            torch.stack(spiraled)
            if isinstance(spiraled[0], torch.Tensor)
            else np.stack(spiraled)
        )
        
    def antispiralise(self, galaxy):
        """
        Change ViT patch ordering from spiral to raster order. See 'spiralise'.
        """
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            self._spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        antispiraled = [galaxy[ii] for ii in indices]
        return (
            torch.stack(antispiraled)
            if isinstance(antispiraled[0], torch.Tensor)
            else np.stack(antispiraled)
        )
        
    def process_galaxy(self, raw_galaxy):
        patch_size = self.modality_registry.get_config("images").patch_size
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        if "images" in self.transform:
            patch_galaxy = self.transform["images"](patch_galaxy)
        if self.spiral:
            patch_galaxy = self.spiralise(patch_galaxy)

        return patch_galaxy
    
    def process_spectra(self, raw_spectra, wavelength):
        patch_size = self.modality_registry.get_config("spectra").patch_size
        # Apply padding to the spectrum
        w = raw_spectra.shape[0]
        window = patch_size
        pad_w = (window - w % window) % window
        padded_spectra = F.pad(raw_spectra, (0, pad_w))
        padded_wl = F.pad(wavelength, (0, pad_w))

        # Now rearrange into patches
        patch_spectra = einops.rearrange(
            padded_spectra,
            "(w p) -> (w) (p)",
            p=window,
        )

        patch_wl = einops.rearrange(
            padded_wl,
            "(w p) -> (w) (p)",
            p=window,
        )

        if "spectra" in self.transform:
            patch_spectra = self.transform["spectra"](patch_spectra)

        return patch_spectra, patch_wl

    @staticmethod
    def process_modes(x, modality_registry, device, shuf=False):
        """Move all tensor values in dictionary x to the specified device.
        And split into X and Y according to the modality registry."""
        modes = modality_registry.generate_sequence(shuf=shuf)

        # Move all tensors to device first
        x_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()
        }

        X = {}
        Y = {}
        for ii, mode in enumerate(modes):
            X[mode] = x_on_device[mode]
            X[f"{mode}_positions"] = x_on_device[f"{mode}_positions"]
            Y[mode] = x_on_device[mode]
            if ii == 0:
                Y[mode] = Y[mode][:, 1:]
            if len(modes) == 1:
                X[mode] = X[mode][:, :-1]
                X[f"{mode}_positions"] = X[f"{mode}_positions"][:, :-1]

        return {"X": X, "Y": Y}

    def _load_fits_data(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Missing file: {path}")
            return None
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
            #print(f"[INFO] Loaded image: {path}")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            return None

    def _load_desi_spectrum(self, healpix_id, targetid, survey, program):
        # Construct filename
        spectrum_filename = f"coadd-{survey.lower()}-{program.lower()}-{healpix_id}.fits"

        # Choose subfolder based on survey
        if survey.lower() == "main":
            #spectra_folder = "/home/valonso/iac18_aasensio_shared/desi_dr1_main"
            spectra_folder = "/home/valonso/iac18_aasensio_shared/desi_dr1_main_debug"
            
        elif survey.lower() == "sv3":
            spectra_folder = "/home/valonso/iac18_aasensio_shared/desi_dr1_sv3/"

        else:
            print(f"[WARNING] Survey {survey} not supported")
            return None

        print(spectra_folder)
        spectrum_path = os.path.join(spectra_folder, spectrum_filename)

        #print(f"[INFO] Looking for spectrum: {spectrum_path}")
        if not os.path.exists(spectrum_path):
            print(f"[WARNING] Spectrum file not found: {spectrum_path}")
            return None

        try:
            spectra = desispec.io.read_spectra(spectrum_path, skip_hdus=['EXP_FIBERMAP', 'SCORES', 'EXTRA_CATALOG', 'MASK', 'RESOLUTION'])
            #print(f"[INFO] Loaded spectra file with {len(spectra.target_ids())} targets")

            if targetid not in spectra.target_ids():
                print(f"[WARNING] TARGETID {targetid} not found in {spectrum_filename}")
                return None

            selected = spectra.select(targets=[targetid])
            combined = coaddition.coadd_cameras(selected)

            reorder_idx = find_matching_indices([targetid], combined.target_ids())

            wave = combined.wave["brz"].astype(np.float32)
            flux = combined.flux["brz"][reorder_idx].astype(np.float32)[0]
            ivar = combined.ivar["brz"][reorder_idx].astype(np.float32)[0]
            #mask = combined.mask["brz"][reorder_idx].astype(np.uint32)[0]
            #res = combined.resolution_data["brz"][reorder_idx].astype(np.float32)[0]

            #print(f"[INFO] Successfully loaded spectrum for TARGETID {targetid}")

            return {
                "wavelength": wave,
                "flux": flux,
                "ivar": ivar,
                #"mask": mask,
                #"res": res,
                "targetid": combined.target_ids()[reorder_idx][0]
            }

        except Exception as e:
            print(f"[ERROR] Failed to load spectrum for TARGETID {targetid}: {e}")
            return None

    def plot_entry(self, idx, save_path=None):
        entry = self.meta[idx]
        data = self[idx]
        segmentation_area = entry.get('segmentation_area', None)

        # UV/AGN lines (rest-frame, in Angstrom)
        main_lines = {
            "Lyα": 1216,
            "C IV": 1549,
            "C III]": 1909,
            "Mg II": 2798,
            "[O II]": 3727,
            "[Ne III]": 3869,
            "Hδ": 4102,
            "Hγ": 4341,
            "Hβ": 4861,
            "[O III]": 4959,
            "[O III]": 5007,
            "[N II]": 6548,
            "Hα": 6563,
            "[N II]": 6584,
            "[S II]": 6717,
            "[S II]": 6731
        }
        redshift = entry.get('Z', 0.0)

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 5)

        # Spectrum plot
        ax_spec = fig.add_subplot(gs[:, :2])
        spec = data["SPECTRUM"]
        if spec is not None:
            wave = spec["wavelength"]
            flux = spec["flux"]

            # Smooth the flux (sigma=2 can be adjusted)
            smoothed_flux = scipy.ndimage.gaussian_filter1d(flux, sigma=5)

            ax_spec.plot(wave, smoothed_flux, label="Smoothed Spectrum", color='black')
            ax_spec.set_xlabel("Observed Wavelength [Å]")
            ax_spec.set_ylabel("Flux")
            ax_spec.set_title(f"Spectrum (TARGETID={data['TARGETID']}, z={redshift:.3f})")

            # Mark main UV lines at observed positions
            for name, rest_wave in main_lines.items():
                obs_wave = rest_wave * (1 + redshift)
                # Only mark if within the observed wavelength range
                if wave.min() < obs_wave < wave.max():
                    ax_spec.axvline(obs_wave, color='red', linestyle='--', alpha=0.7)
                    ax_spec.text(obs_wave, ax_spec.get_ylim()[1]*0.9, name, rotation=90, color='red', va='top', ha='center', fontsize=8)
            # Set axis limits to observed spectrum range
            ax_spec.set_xlim(wave.min(), wave.max())
        else:
            ax_spec.text(0.5, 0.5, "No spectrum", ha='center', va='center', transform=ax_spec.transAxes)

        # VIS cutout
        ax_vis = fig.add_subplot(gs[0, 2])
        vis_img = data["VIS"]
        if vis_img is not None:
            ax_vis.imshow(vis_img, cmap='gray', origin='lower')
            ax_vis.set_title("VIS")
            ax_vis.axis('off')
            if segmentation_area is not None:
                radius = np.sqrt(segmentation_area / np.pi) + 10  # Make circle bigger
                h, w = vis_img.shape
                circ = Circle((w/2, h/2), radius, edgecolor='lime', facecolor='none', lw=2, alpha=0.8)
                ax_vis.add_patch(circ)
        else:
            ax_vis.text(0.5, 0.5, "No VIS", ha='center', va='center', transform=ax_vis.transAxes)
            ax_vis.axis('off')

        # New composite (top-center, gs[0,3]): Y-J-H RGB (R=H, G=J, B=Y)
        ax_comp_rgb = fig.add_subplot(gs[0, 3])
        composite_rgb = make_nisp_rgb_composite(data.get("NISP", {}), band_order=('H', 'J', 'Y'))
        if composite_rgb is not None:
            ax_comp_rgb.imshow(composite_rgb, origin='lower')
            ax_comp_rgb.set_title("NISP RGB (R=H, G=J, B=Y)")
            ax_comp_rgb.axis('off')
            if segmentation_area is not None:
                h, w = composite_rgb.shape[:2]
                radius = np.sqrt(segmentation_area / np.pi) + 10
                circ = Circle((w/2, h/2), radius, edgecolor='lime', facecolor='none', lw=2, alpha=0.8)
                ax_comp_rgb.add_patch(circ)
        else:
            ax_comp_rgb.text(0.5, 0.5, "No NISP RGB", ha='center', va='center', transform=ax_comp_rgb.transAxes)
            ax_comp_rgb.axis('off')

        # Composite cutout (top-right, gs[0,4]) -- unchanged behavior
        ax_comp = fig.add_subplot(gs[0, 4])
        composite = make_composite_cutout(vis_img, data.get("NISP", {}))
        if composite is not None:
            ax_comp.imshow(composite, origin='lower')
            ax_comp.set_title("Composite (NISP, mean, VIS)")
            ax_comp.axis('off')
            if segmentation_area is not None:
                # segmentation circle only if shapes match (composite made)
                h, w = composite.shape[:2]
                radius = np.sqrt(segmentation_area / np.pi) + 10
                circ = Circle((w/2, h/2), radius, edgecolor='lime', facecolor='none', lw=2, alpha=0.8)
                ax_comp.add_patch(circ)
        else:
            ax_comp.text(0.5, 0.5, "No composite", ha='center', va='center', transform=ax_comp.transAxes)
            ax_comp.axis('off')

        # NISP cutouts
        for i, band in enumerate(['Y', 'J', 'H']):
            ax_nisp = fig.add_subplot(gs[1, 2 + i])
            nisp_img = data["NISP"].get(band)
            if nisp_img is not None:
                ax_nisp.imshow(nisp_img, cmap='gray', origin='lower')
                ax_nisp.set_title(f"NISP-{band}")
                ax_nisp.axis('off')
                if segmentation_area is not None:
                    radius = np.sqrt(segmentation_area / np.pi) + 10
                    h, w = nisp_img.shape
                    circ = Circle((w/2, h/2), radius, edgecolor='lime', facecolor='none', lw=2, alpha=0.8)
                    ax_nisp.add_patch(circ)
            else:
                ax_nisp.text(0.5, 0.5, f"No NISP-{band}", ha='center', va='center', transform=ax_nisp.transAxes)
                ax_nisp.axis('off')

        # adjust spacing so top composites don't overlap NISP titles
        fig.subplots_adjust(top=0.92, hspace=0.75, wspace=0.35)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[INFO] Figure saved to {save_path}")
        plt.show()
    
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.stochastic:
            idx = np.random.randint(self.dataset_len)   
            
        if not self.stochastic and idx >= len(self.meta):
            raise IndexError(f"Debug index {idx} out of reange for a dataset with len {len(self.meta)}")

        # From Gosia's datalaoder
        entry = self.meta[idx]
        targetid = entry['TARGETID']
        healpix_id = entry['HEALPIX']
        survey = entry['SURVEY']
        program = entry['PROGRAM']
        vis_filename = entry['name']

        # Load VIS image
        vis_image_path = os.path.join(self.vis_folder, vis_filename)
        vis_image = self._load_fits_data(vis_image_path)

        if vis_image is None:
            raise FileNotFoundError(f"VIS image {vis_filename} (idx {idx}) not found. Skipping.")

        # Load NISP images (H, J, Y)
        ref_shape = vis_image.shape
        nisp_images_list = []
        for band in ['H', 'J', 'Y']:
            nisp_filename = vis_filename.replace("VIS", f"NIR-{band}")
            
            nisp_path = os.path.join(self.nisp_folders[band], nisp_filename)
            nisp_img = self._load_fits_data(nisp_path)
            
            # Managing not found NISP images
            if nisp_img is None or nisp_img.shape != ref_shape:
                nisp_images_list.append(np.zeros_like(vis_image))
            # Replacing image by zeros array
            else:
                nisp_images_list.append(nisp_img)

        # Staking images: [VIS, NISP-H, NISP-J, NISP-Y]
        raw_galaxy = np.stack([vis_image] + nisp_images_list, axis=0).astype(np.float32)
        raw_galaxy = torch.tensor(raw_galaxy).to(torch.bfloat16)

        # Loading spectrum as Gosia's dataloader
        #spectrum_dict = self._load_desi_spectrum(healpix_id, targetid, survey, program)
        #if spectrum_dict is None:
        #    raise FileNotFoundError(f"Spectrum for TARGETID {targetid} (idx {idx}) not found. Skipping.")
        
        # Loading preprocces spectra
        original_path = entry['SPEC_PATH']
        spectrum_filename = os.path.basename(original_path)
        spectrum_path = os.path.join(self.spectra_folder, spectrum_filename)   

        try:
            with fits.open(spectrum_path) as hdul:
                data = hdul[1].data 
                wave = data['WAVELENGTH'].astype(np.float32)
                flux = data['FLUX'].astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"[ERROR] Unable to read spectra: {spectrum_path}. Error: {e}")
        
        # Converting spectrum to tensor
        raw_spectra = torch.tensor(flux).to(torch.bfloat16)
        wavelength = torch.tensor(wave).to(torch.bfloat16)
        
        # Normalizing wavelength
        wavelength = (wavelength - 3000) / (10000 - 3000) # Normalizar

        # Patching the galaxy
        patch_galaxy = self.process_galaxy(raw_galaxy)
        patch_spectra, patch_wl = self.process_spectra(raw_spectra, wavelength)

        # Checking incorrect data fot NaNs and torch
        if torch.isnan(patch_galaxy).any():
            raise ValueError(f"NaNs found in image (idx {idx}). Skipping.")
        if torch.isnan(patch_spectra).any() or torch.isnan(patch_wl).any():
            raise ValueError(f"NaNs found in spectrum (idx {idx}). Skipping.")
        
        # Formatting output as expected by AstroPT
        return {
            "images": patch_galaxy,
            "images_positions": torch.arange(0, len(patch_galaxy), dtype=torch.long),
            "spectra": patch_spectra,
            "spectra_positions": patch_wl,
            "idx": idx,
        }


def adjust_dynamic_range(flux, q=100, clip=99.85):
    """Adjust dynamic range using asinh scaling and optional percentile clipping."""
    im = np.arcsinh(flux * q)
    if clip < 100:
        im = np.clip(im, 0, np.percentile(im, clip))
    return im


# new helper to test whether an image is usable
def is_valid_image(im, min_range=1e-6):
    """Return False if im is None, has only NaNs, or has negligible dynamic range."""
    if im is None:
        return False
    if not np.isfinite(im).any():
        return False
    try:
        im_min = np.nanmin(im)
        im_max = np.nanmax(im)
    except Exception:
        return False
    if not np.isfinite(im_min) or not np.isfinite(im_max):
        return False
    return (im_max - im_min) > min_range


def to_uint8(im, clip_below_zero=True):
    """Normalize to [0,255] uint8. If image is constant/invalid, return mid-gray."""
    if im is None:
        return None
    if clip_below_zero:
        im = np.clip(im, 0, None)
    # handle all-NaN or constant arrays gracefully
    if not np.isfinite(im).any():
        # return neutral gray image
        shape = im.shape
        return np.full(shape, 128, dtype=np.uint8)
    im_min = np.nanmin(im)
    im_max = np.nanmax(im)
    if not np.isfinite(im_min) or not np.isfinite(im_max) or im_max <= im_min:
        # constant image -> neutral gray
        shape = im.shape
        return np.full(shape, 128, dtype=np.uint8)
    im = (im - im_min) / (im_max - im_min)
    return (255 * im).astype(np.uint8)


def make_composite_cutout(vis_cutout, nisp_cutouts, vis_q=100, vis_clip=99.85, nisp_q=1, nisp_clip=99.85):
    """
    Create a composite RGB cutout from VIS and NISP images.

    - vis_cutout: 2D array or None
    - nisp_cutouts: dict of band->2D array (e.g. {'Y': arr, 'J': arr, 'H': arr})
    Returns an HxWx3 uint8 image or None if not enough data / shape mismatch.
    """
    logging.debug('Creating composite cutout')

    # require a valid VIS image
    if not is_valid_image(vis_cutout):
        logging.debug('VIS cutout missing or invalid, cannot create composite')
        return None

    # collect available and valid NISP images (filter out empty/constant)
    raw_available = [img for img in (nisp_cutouts or {}).values() if img is not None]
    available = [img for img in raw_available if is_valid_image(img)]
    if len(available) == 0:
        logging.debug('No valid NISP bands available, cannot create composite')
        return None

    # ensure shapes match
    for arr in available:
        if arr.shape != vis_cutout.shape:
            logging.debug(f'vis shape {vis_cutout.shape}, nisp shape {arr.shape} - cannot make composite')
            return None

    # mean of available NISP bands
    nisp_mean = np.mean(np.stack(available, axis=0), axis=0)

    # adjust dynamic ranges
    vis_adj = adjust_dynamic_range(vis_cutout, q=vis_q, clip=vis_clip)
    nisp_adj = adjust_dynamic_range(nisp_mean, q=nisp_q, clip=nisp_clip)
    mean_flux = np.mean([vis_adj, nisp_adj], axis=0)

    # convert to uint8 (to_uint8 now returns neutral gray for constant/invalid arrays)
    vis_uint8 = to_uint8(vis_adj)
    nisp_uint8 = to_uint8(nisp_adj)
    mean_uint8 = to_uint8(mean_flux)

    im = np.stack([nisp_uint8, mean_uint8, vis_uint8], axis=2)
    return im

# new helper: build RGB from specified NISP bands (use neutral gray for missing bands)
def make_nisp_rgb_composite(nisp_cutouts, band_order=('H', 'J', 'Y'), q=1, clip=99.85):
    """
    Build RGB image from NISP bands in given order.
    band_order: tuple/list where first element maps to R, second->G, third->B.
    Missing bands are substituted with neutral gray. Returns HxWx3 uint8 or None.
    """
    logging.debug('Creating NISP RGB composite with bands: %s', band_order)
    if not nisp_cutouts:
        logging.debug('No NISP inputs')
        return None

    # collect band arrays in requested order
    bands = [nisp_cutouts.get(b) for b in band_order]
    # require at least one valid band
    valid_any = any(is_valid_image(b) for b in bands if b is not None)
    if not valid_any:
        logging.debug('No valid NISP bands present for RGB composite')
        return None

    # determine reference shape from first valid band
    ref = None
    for b in bands:
        if b is not None and is_valid_image(b):
            ref = b.shape
            break
    if ref is None:
        return None

    # ensure shapes of valid bands match ref
    for b in bands:
        if b is not None and is_valid_image(b) and b.shape != ref:
            logging.debug('Shape mismatch among NISP bands; abort RGB composite')
            return None

    # prepare channels: if band valid -> adjust & convert, else neutral gray
    channels = []
    for b in bands:
        if b is not None and is_valid_image(b):
            b_adj = adjust_dynamic_range(b, q=q, clip=clip)
            ch = to_uint8(b_adj)
        else:
            ch = np.full(ref, 128, dtype=np.uint8)
        channels.append(ch)

    # stack as R,G,B where band_order maps to R,G,B
    rgb = np.stack(channels, axis=2)
    return rgb




