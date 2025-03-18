import numpy as np
import math
from astropy.io import fits
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


def write_results_to_file(metrics, parameter):

    file_name = f"{parameter}_Euclid_NNPZ.txt"

    # Write to the file
    with open(file_name, 'w') as txt_file:
        txt_file.write("Final Evaluation Statistics\n")
        txt_file.write("===========================\n")
        txt_file.write("Percentage: 100\n")
        # Write mean and std metrics
        for key in metrics:
            txt_file.write(f"{key}: {metrics[key]:.4f}\n")
            

def compute_catalog_statistics(catalog_path, parameter):
    with fits.open(catalog_path) as hdul:
        catalog_data = hdul[1].data


    if parameter == 'PhotoZ':
        specz = catalog_data['Z'] # DESI spec-z
        photoz = catalog_data['phz_median'] # NNPZ photo-z
        valid_indices = (
            ~np.isnan(specz) &
            ~np.isnan(photoz)         
        )        
    elif parameter == 'logM':
        specz = catalog_data['LOGM'] # DESI stellar mass
        photoz = catalog_data['phz_pp_median_stellarmass'] # NNPZ stellar mass
        valid_indices = (
            ~np.isnan(specz) &
            ~np.isnan(photoz) &
            (catalog_data['spurious_flag'] == 0) &
            (catalog_data['det_quality_flag'] < 4) &
            (catalog_data['mumax_minus_mag'] > -2.6) &
            (catalog_data['LOGM'] > 0) &
            (catalog_data['CHI2'] < 17) &
            (catalog_data['LOGM_ERR'] < 0.25) &
            (catalog_data['phz_flags'] == 0)          
        )                 
    elif parameter == 'logM_Enia': #Stellar masses from Enia et al. 2025 (including IRAC bands)
        specz = catalog_data['LOGM']
        photoz = catalog_data['opp_median_stellarmass']
        valid_indices = (
            ~np.isnan(specz) &
            ~np.isnan(photoz) &
            (catalog_data['spurious_flag'] == 0) &
            (catalog_data['det_quality_flag'] < 4) &
            (catalog_data['mumax_minus_mag'] > -2.6) &
            (catalog_data['phz_flags'] == 0)          
        ) 
    else:
        print("The parameter is not given")


    valid_specz = specz[valid_indices]
    valid_photoz = photoz[valid_indices]
    print(len(valid_specz))  
    print(len(valid_photoz)) 
    valid_specz = np.asarray(valid_specz, dtype=float)
    valid_photoz = np.asarray(valid_photoz, dtype=float)
  
    # Plot the true vs predicted redshifts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(22, 10)) 
    plt.subplots_adjust(left=0.18, bottom=None, right=0.98, top=None, wspace=0.02, hspace=0)

    ax = sns.kdeplot(x=valid_specz, y=valid_photoz, cmap="RdYlBu_r", fill=True, 
                     levels=15, thresh=0.05)  
                     
    plt.gca().set_facecolor('white')

    # Add the y=x line
    max_val = max(valid_specz.max(), valid_photoz.max())
    min_val = min(valid_specz.min(), valid_photoz.min())
    if parameter =="PhotoZ":
            plt.plot([-0.5,1.1], [-0.5,1.1], '--', linewidth=3, color='black')
            plt.plot([-0.5, 1.1], [-0.5 + 0.15, 1.1 + 0.15], ':', linewidth=3, color='black')  
            plt.plot([-0.5, 1.1], [-0.5 - 0.15, 1.1 - 0.15], ':', linewidth=3, color='black')  
    else:
            plt.plot([0.5*min_val, 1.5*max_val], [0.5*min_val, 1.5*max_val], '--', linewidth=3, color='black')
            plt.plot([0.5 * min_val, 1.5 * max_val], [0.5 * min_val + 0.25, 1.5 * max_val + 0.25], ':', linewidth=3, color='black')  
            plt.plot([0.5 * min_val, 1.5 * max_val], [0.5 * min_val - 0.25, 1.5 * max_val - 0.25], ':', linewidth=3, color='black') 

    #plt.text(
    #0.15, 0.75, r'$\rm \eta_{out} = %.2f \pm %.2f\%%$' % (outlier_fraction, outlier_fraction_err),
    #horizontalalignment='left', verticalalignment='bottom',
    #fontsize=55, transform=plt.gca().transAxes
    #)


    if parameter == 'PhotoZ':
        plt.xlabel(r'$z_{\mathrm{DESI}}$', fontsize=60)
        plt.ylabel(r'$\mathrm{photo-}z_{{\tt NNPZ}}$', fontsize=60)
    elif parameter == 'logM':
        plt.xlabel(r"$\log_{10} (M_*/M_{\odot})_{\rm DESI}$", fontsize=60)
        plt.ylabel(r"$\log_{10} (M_*/M_{\odot})_{\tt NNPZ}$", fontsize=60)
    elif parameter == 'logM_Enia':
        plt.xlabel(r'$\mathrm{True \, z \, log(M_{star}/M_{\odot})_{DESI}}$', fontsize=60)
        plt.ylabel(r"$\rm Euclid\mbox{ } log(M_{star}/M_{\odot})_{IRAC}$", fontsize=60)

    # Adjust ticks and formatting
    plt.minorticks_on()
    plt.tick_params(axis='x', which='major', labelsize=55)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=55)  
    plt.tick_params(axis='both', which='major', length=15, width=2, direction='in', labelsize=55)  
    plt.tick_params(axis='both', which='minor', length=10, width=2, direction='in', labelsize=55)  

    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    if parameter == 'PhotoZ':
        plt.xlim(-0.18,1.1)
        plt.ylim(-0.18,1.1)        
    elif parameter == 'logM' :
        plt.xlim(6.8, 13)
        plt.ylim(6.8, 13)    
    elif parameter == 'logM_Enia':
        plt.xlim(5, 14)
        plt.ylim(5, 14)  
    else:
        plt.xlim(0.5*min_val, 1.5*max_val)
        plt.ylim(0.5*min_val, 1.5*max_val)        
        

    # Remove grid and legend, set white background
    plt.gca().set_facecolor('white')
    plt.grid(False)
    plt.legend([], [], frameon=False)


    # Save the plot
    plt.tight_layout()
    plot_file = f"{parameter}_Euclid_NNPZ.png"
    plt.savefig(plot_file)
    plt.close()

    # Compute metrics
    mse = mean_squared_error(valid_specz, valid_photoz)
    mae = mean_absolute_error(valid_specz, valid_photoz)
    r2 = r2_score(valid_specz, valid_photoz)

    delta_z = (valid_photoz - valid_specz) / (1 + valid_specz)
    mean_delta_z = np.mean(delta_z)
    sigma_68 = (np.percentile(delta_z, 84.1) - np.percentile(delta_z, 15.9)) / 2
    outlier_fraction = 100 * len(valid_photoz[np.abs(delta_z) >= 0.15]) / len(valid_photoz)
    nmad = 1.48 * np.median(np.abs(delta_z - np.median(delta_z)))


    print("Evaluation Statistics")
    print("=====================")
    print(f"Number of valid sources: {len(valid_specz)}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Bias: {mean_delta_z:.4f}")
    print(f"Sigma 68: {sigma_68:.4f}")
    print(f"Outlier Fraction (|Δz| ≥ 0.15): {outlier_fraction:.2f}%")
    print(f"NMAD: {nmad:.4f}")
    
    return {
    "mse": mse,
    "mae": mae,
    "r2": r2,
    "bias": mean_delta_z,
    "nmad": nmad,
    "sigma_68": sigma_68,
    "outlier_fraction": outlier_fraction,
    }

# Path to the catalog file
parameter = 'PhotoZ' # set PhotoZ or logM
catalog_path = "../../../Q1_data/EuclidMorphPhysPropSpecZ.fits"

    
metrics = compute_catalog_statistics(catalog_path, parameter)
write_results_to_file(metrics,parameter)
