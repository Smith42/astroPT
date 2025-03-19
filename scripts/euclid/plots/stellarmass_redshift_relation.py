import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load a sample with cuts
def load_your_sample(catalog_path):
    with fits.open(catalog_path) as hdul:
        catalog_data = hdul[1].data
    
    # Apply cuts
    mask = (
        ~np.isnan(catalog_data['phz_median']) &
        (catalog_data['spurious_flag'] == 0) &
        (catalog_data['det_quality_flag'] < 4) &
        (catalog_data['mumax_minus_mag'] > -2.6) &
        (catalog_data['phz_flags'] == 0)
    )
    
    my_sample = {
        'redshift': catalog_data['phz_median'][mask],
        'stellar_mass': catalog_data['phz_pp_median_stellarmass'][mask],
    }
    
    return my_sample

# Load the entire Q1 sample
def load_q1_sample(q1_path):
    with fits.open(q1_path) as hdul:
        q1_data = hdul[1].data

    # Apply cuts
    mask = (
        ~np.isnan(q1_data['phz_median']) &
        (q1_data['spurious_flag'] == 0) &
        (q1_data['det_quality_flag'] < 4) &
        (q1_data['mumax_minus_mag'] > -2.6) &
        (q1_data['phz_flags'] == 0)
    )
        
    q1_sample = {
        'redshift': q1_data['phz_median'][mask],
        'stellar_mass': q1_data['phz_pp_median_stellarmass'][mask]
    }
    
    return q1_sample

# Load the DESI sample
def load_desi_sample(desi_path):
    with fits.open(desi_path) as hdul:
        desi_data = hdul[1].data

    mask = (
        ~np.isnan(desi_data['phz_median']) &
        (desi_data['spurious_flag'] == 0) &
        (desi_data['det_quality_flag'] < 4) &
        (desi_data['mumax_minus_mag'] > -2.6) &
        (desi_data['phz_flags'] == 0) &
        (desi_data['LOGM'] > 0) &
        (desi_data['CHI2'] <= 17) 
    )
    
    desi_sample = {
        'phz_median': desi_data['phz_median'][mask],
        'phz_pp_median_stellarmass': desi_data['phz_pp_median_stellarmass'][mask],
        'redshift': desi_data['Z'][mask],
        'stellar_mass': desi_data['LOGM'][mask]
    }
    
    return desi_sample

# Plot stellar mass vs redshift with residual insets
def plot_stellar_mass_redshift_with_residuals(your_sample, q1_sample, desi_sample):
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.gca()  # Get the current axis
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=10, width=1.5, direction='in', labelsize=20)
    ax.tick_params(axis='both', which='minor', length=6, width=1.5, direction='in')
    ax.tick_params(top=True, bottom=True, left=True, right=True)  # Enable ticks on all four sides
    
    # Plot Q1 sample as a grid (hexbin)
    hb = ax.hexbin(q1_sample['redshift'], q1_sample['stellar_mass'], 
                   gridsize=100, cmap='Blues', bins='log', alpha=0.6, label=r'$\rm Q1\mbox{ } photo-z_{NNPZ}$')
    
    # Plot your sample
    ax.scatter(your_sample['redshift'], your_sample['stellar_mass'], 
               color='blue', label=r'$\rm This\mbox{ } work \mbox{ }photo-z_{NNPZ}$', alpha=0.4, edgecolor='black')
    
    # Plot DESI sample
    ax.scatter(desi_sample['redshift'], desi_sample['stellar_mass'], 
               color='red', label=r'DESI spec-$z$', alpha=0.3, edgecolor='black')
    
    # Set axis labels and limits
    ax.set_xlabel(r"$z$", fontsize=24)
    ax.set_ylabel(r"$\rm log(M_{star}/M_{\odot})$", fontsize=24)
    ax.set_ylim(3, 15)  # Set stellar mass range
    ax.set_xlim(-0.01, 3)  # Set redshift range
    
    # Add legend
    ax.legend(fontsize=16, frameon=False, loc='lower right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add residual insets
    # Redshift residuals
    redshift_residuals = desi_sample['phz_median'] - desi_sample['redshift']
    ax_inset_z = ax.inset_axes([0.8, 0.8, 0.2, 0.2])  # Position: [x, y, width, height]
    counts, bins = np.histogram(redshift_residuals, bins=300, density=False)  # density=False to get raw counts
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts = counts.astype(np.float64)
    
    counts /= np.max(counts) 
    ax_inset_z.plot(bin_centers, counts, color='black', alpha=1.0)
    ax_inset_z.set_xlabel(r"$\rm \Delta z$", fontsize=12)
    ax_inset_z.tick_params(labelsize=10)
    ax_inset_z.grid(True, linestyle='--', alpha=0.6)
    ax_inset_z.set_xlim(-0.2, 0.2)
    ax_inset_z.axvline(x=0,linestyle='-',color='black')
    ax_inset_z.axvline(x=0,linestyle='-',color='black')
    ax.tick_params(axis='both', which='major', length=10, width=1.5, direction='in', labelsize=20)
    ax.tick_params(top=True, bottom=True, left=True, right=True)  # Enable ticks on all four sides    
    
    # Stellar mass residuals
    stellar_mass_residuals = desi_sample['phz_pp_median_stellarmass'] - desi_sample['stellar_mass']
    ax_inset_m = ax.inset_axes([0.595, 0.8, 0.2, 0.2])  # Position: [x, y, width, height]
    counts, bins = np.histogram(stellar_mass_residuals, bins=200, density=False)  # density=False to get raw counts
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts = counts.astype(np.float64)    
    counts /= np.max(counts) 
    ax_inset_m.plot(bin_centers, counts, color='black', alpha=1.0)
    #ax_inset_m.hist(stellar_mass_residuals, bins=50, color='black', alpha=1.0, density=True, histtype='step')
    ax_inset_m.set_xlabel(r"$\rm \Delta log(M_{star}/M_{\odot})$", fontsize=12)
    ax_inset_m.tick_params(labelsize=10)
    ax_inset_m.grid(True, linestyle='--', alpha=0.6)
    ax_inset_m.set_xlim(-1., 1.)
    ax_inset_m.axvline(x=0,linestyle='-',color='black')
    
    plt.savefig("stellarmass_redshift.png")
        
    plt.show()
    

# Main function
def main():
    # File paths
    catalog_path = "../../Q1_data/EuclidMorphPhysPropSpecZ.fits"
    q1_path = "../../Q1_data/raw_files/euclid_bars_sersic-Q1.fits"
    desi_path = "../../Q1_data/DESI_logM.fits"
    
    # Load samples
    your_sample = load_your_sample(catalog_path)
    q1_sample = load_q1_sample(q1_path)
    desi_sample = load_desi_sample(desi_path)
    
    # Plot stellar mass vs redshift with residuals
    plot_stellar_mass_redshift_with_residuals(your_sample, q1_sample, desi_sample)

if __name__ == "__main__":
    main()
