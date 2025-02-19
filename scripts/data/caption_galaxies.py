import os
from datasets import load_dataset, concatenate_datasets
from itertools import chain
from google import genai
from tqdm import tqdm

# Configure Google API key
os.environ["GOOGLE_API_KEY"] = KEY
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def create_galaxy_prompt(example):
    """
    Convert galaxy metadata to a comprehensive structured prompt for an LLM.
    
    Args:
        example: Dictionary containing galaxy metadata
    
    Returns:
        str: Formatted prompt with all relevant galaxy information
    """
    # Base prompt
    base_prompt = """
Describe this image of a galaxy in detail. Only describe the galaxy, not the foreground objects surrounding it. Act like a citizen scientist. Do not converse please, just give your description in a scientific tone.

Here is additional information about this galaxy:
"""
    
    # Helper function to format values appropriately
    def format_value(key, value):
        # Skip None values, PIL image objects, and index values
        if value is None or str(type(value)).find('PIL') >= 0 or key == '__index_level_0__':
            return None
            
        # Handle numeric values
        if isinstance(value, (int, float)):
            # Format percentages
            if key.endswith('_fraction'):
                return f"{value:.2f}"
            # Format astronomical magnitudes 
            elif key.startswith('mag_') or key.find('_mag_') > 0:
                return f"{value:.2f}"
            # Format coordinates
            elif key in ['ra', 'dec', 'ra_photoz', 'dec_photoz', 'ra_ossy', 'dec_ossy', 'ra_alf', 'dec_alf', 'ra_jhu', 'dec_jhu']:
                return f"{value:.6f}"
            # Format redshifts
            elif key in ['redshift', 'photo_z', 'redshift_nsa', 'redshift_ossy', 'spec_z']:
                return f"{value:.6f}"
            # Format error values
            elif key.find('err') > 0 or key.startswith('e_') or key.startswith('sig'):
                return f"{value:.6f}"
            # Format angular sizes
            elif key.find('theta') > 0 or key.find('phi') > 0:
                return f"{value:.4f}"
            # Format mass and SFR values that are in log scale
            elif (key.startswith('mass_') or key.startswith('sfr_') or key.startswith('ssfr_')) and (-20 < value < 20):
                return f"{value:.4f} (log10)"
            # Default number formatting
            return f"{value:.4f}" if isinstance(value, float) else f"{value}"
            
        # String values
        return value
    
    # Group metadata into categories
    categories = {
        "Basic Information": [
            'dr8_id', 'iauname', 'brickid', 'objid',
            'ra', 'dec', 'photoz_id', 'ra_photoz', 'dec_photoz'
        ],
        
        "Redshift Information": [
            'redshift', 'photo_z', 'photo_zerr', 'spec_z', 'redshift_nsa', 'redshift_ossy'
        ],
        
        "Size and Shape Measurements": [
            'est_petro_th50', 'est_petro_th50_kpc', 'petro_theta', 'petro_th50', 'petro_th90',
            'petro_phi50', 'petro_phi90', 'petro_ba50', 'petro_ba90', 'elpetro_ba', 'elpetro_phi',
            'elpetro_theta_r', 'sersic_n', 'sersic_ba', 'sersic_phi'
        ],
        
        "DESI Magnitudes": [
            'mag_g_desi', 'mag_r_desi', 'mag_z_desi'
        ],
        
        "Legacy Survey Magnitudes": [
            'mag_f', 'mag_n', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'u_minus_r'
        ],
        
        "Absolute Magnitudes": [
            'mag_abs_g_photoz', 'mag_abs_r_photoz', 'mag_abs_z_photoz',
            'elpetro_absmag_f', 'elpetro_absmag_n', 'elpetro_absmag_u', 'elpetro_absmag_g',
            'elpetro_absmag_r', 'elpetro_absmag_i', 'elpetro_absmag_z'
        ],
        
        "Sersic Profile Measurements": [
            'sersic_nmgy_f', 'sersic_nmgy_n', 'sersic_nmgy_u', 'sersic_nmgy_g',
            'sersic_nmgy_r', 'sersic_nmgy_i', 'sersic_nmgy_z'
        ],
        
        "Mass Estimates": [
            'elpetro_mass', 'elpetro_mass_log', 'mass_inf_photoz', 'mass_med_photoz', 'mass_sup_photoz'
        ],
        
        "Star Formation Properties": [
            'sfr_inf_photoz', 'sfr_sup_photoz', 'ssfr_inf_photoz', 'ssfr_med_photoz', 'ssfr_sup_photoz',
            'fibre_sfr_avg', 'fibre_sfr_entropy', 'fibre_sfr_median', 'fibre_sfr_mode',
            'fibre_sfr_p16', 'fibre_sfr_p2p5', 'fibre_sfr_p84', 'fibre_sfr_p97p5',
            'fibre_ssfr_avg', 'fibre_ssfr_entropy', 'fibre_ssfr_median', 'fibre_ssfr_mode',
            'fibre_ssfr_p16', 'fibre_ssfr_p2p5', 'fibre_ssfr_p84', 'fibre_ssfr_p97p5',
            'total_ssfr_avg', 'total_ssfr_entropy', 'total_ssfr_flag', 'total_ssfr_median',
            'total_ssfr_mode', 'total_ssfr_p16', 'total_ssfr_p2p5', 'total_ssfr_p84', 'total_ssfr_p97p5',
            'total_sfr_avg', 'total_sfr_entropy', 'total_sfr_flag', 'total_sfr_median',
            'total_sfr_mode', 'total_sfr_p16', 'total_sfr_p2p5', 'total_sfr_p84', 'total_sfr_p97p5'
        ],
        
        "Morphology (Galaxy Zoo)": [
            'smooth-or-featured_smooth_fraction', 'smooth-or-featured_featured-or-disk_fraction',
            'smooth-or-featured_artifact_fraction', 'disk-edge-on_yes_fraction', 'disk-edge-on_no_fraction',
            'has-spiral-arms_yes_fraction', 'has-spiral-arms_no_fraction', 'bar_strong_fraction',
            'bar_weak_fraction', 'bar_no_fraction', 'bulge-size_dominant_fraction',
            'bulge-size_large_fraction', 'bulge-size_moderate_fraction', 'bulge-size_small_fraction',
            'bulge-size_none_fraction', 'how-rounded_round_fraction', 'how-rounded_in-between_fraction',
            'how-rounded_cigar-shaped_fraction', 'edge-on-bulge_boxy_fraction', 'edge-on-bulge_none_fraction',
            'edge-on-bulge_rounded_fraction', 'spiral-winding_tight_fraction', 'spiral-winding_medium_fraction',
            'spiral-winding_loose_fraction', 'spiral-arm-count_1_fraction', 'spiral-arm-count_2_fraction',
            'spiral-arm-count_3_fraction', 'spiral-arm-count_4_fraction', 'spiral-arm-count_more-than-4_fraction',
            'spiral-arm-count_cant-tell_fraction', 'merging_none_fraction', 'merging_minor-disturbance_fraction',
            'merging_major-disturbance_fraction', 'merging_merger_fraction'
        ],
        
        "OSSY Spectroscopic Properties": [
            'dr7objid_ossy', 'log_l_oiii', 'fwhm', 'e_fwhm', 'equiv_width', 'log_l_ha',
            'log_m_bh', 'upper_e_log_m_bh', 'lower_e_log_m_bh', 'log_bolometric_l'
        ],
        
        "HI Properties": [
            'W50', 'sigW', 'W20', 'HIflux', 'sigflux', 'SNR', 'RMS', 'Dist', 'sigDist', 'logMH', 'siglogMH'
        ],
        
        "Other Measurements": [
            'sky_separation_arcsec_from_photoz', 'elpetro_flux_r'
        ]
    }
    
    # Build the prompt with organized scientific data
    formatted_prompt = base_prompt
    
    for category, keys in categories.items():
        # Check if we have any values for this category
        category_values = {}
        for key in keys:
            if key in example and example[key] is not None:
                formatted_value = format_value(key, example[key])
                if formatted_value is not None:
                    category_values[key] = formatted_value
        
        # If we have values, add the category
        if category_values:
            formatted_prompt += f"\n\n{category}:"
            for key, value in category_values.items():
                # Format key for better readability
                display_key = key.replace('_', ' ').replace('-', ' ')
                display_key = ' '.join(word.capitalize() for word in display_key.split())
                formatted_prompt += f"\n- {display_key}: {value}"
    
    # Add a final instruction
    formatted_prompt += """

Based on this information and what you see in the image, provide a detailed scientific description of this galaxy. Do not include references to the values above, simply use them to roughly guide your description.
"""
    
    return formatted_prompt

def caption_image(image, information):
    """Generate caption for an image using Gemini Flash 2.0"""
    # Prepare the prompt
    prompt = create_galaxy_prompt(information)
    
    # Generate response from Gemini
    response = client.models.generate_content(
        contents=[prompt, image],
        model="gemini-2.0-flash",
    )
    
    return response.text

def main():
    galaxies = load_dataset("Smith42/galaxies", split="test", name="with_crops", revision="refs/pr/2", streaming=True)
    metadata = load_dataset("Smith42/galaxies_metadata", split="test", streaming=True).remove_columns("dr8_id")
    dataset = concatenate_datasets([galaxies, metadata], axis=1)
    
    results = []
    for i, example in tqdm(enumerate(dataset), total=4):
        if i >= 4:
            break
            
        try:
            # Get image from dataset
            # Adjust the key based on actual dataset structure
            image = example['image_crop']
            image_id = example['dr8_id']
            
            # Generate caption
            caption = caption_image(image, example)
            
            # Store results
            results.append({
                'dr8_id': image_id,
                'caption': caption
            })
            
            # Optional: save results periodically
            #if i > 0 and i % 10 == 0:
            #    save_results(results, f"galaxy_captions_batch_{i//10}.json")
                
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
    
    # Save final results
    save_results(results, "gemini2.json")
    print(f"Completed captioning {len(results)} galaxy images")

def save_results(results, filename):
    """Save captioning results to JSON file"""
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
