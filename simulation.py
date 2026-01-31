"""
Strong Lensing Simulation with PyAutoLens

This script generates simulated strong lensing images for three categories:
1. No substructure
2. CDM subhalos
3. Superfluid dark matter vortices

These images are designed for training machine learning models.
"""

import numpy as np
import autolens as al
import autolens.plot as aplt
from pathlib import Path
import json
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from astropy import units
from scipy.ndimage import gaussian_filter


class StrongLensingSimulator:
    """Simulate strong lensing images with different dark matter models."""
    
    def __init__(self, 
                 output_dir: str = "subhalo_dataset",
                 grid_size: int = 100,
                 pixel_scale: float = 0.05,
                 seed: int = None):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        output_dir : str
            Directory to save generated images and labels
        grid_size : int
            Number of pixels in each dimension
        pixel_scale : float
            Size of each pixel in arcseconds
        seed : int
            Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.grid_size = grid_size
        self.pixel_scale = pixel_scale
        
        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Create grid for image generation
        self.grid = al.Grid2D.uniform(
            shape_native=(grid_size, grid_size),
            pixel_scales=pixel_scale
        )
        
    def sample_dm_halo(self) -> al.mp.Isothermal:
        """
        Create the main dark matter halo (lens galaxy).
        Fixed parameters as specified.
        """
        return al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,  # Derived from M_tot=1e12
            ell_comps=(0.0, 0.0)
        )
    
    def sample_external_shear(self) -> al.mp.ExternalShear:
        """
        Sample external shear parameters.
        
        Returns
        -------
        ExternalShear profile with:
        - γext: uniform [0.0, 0.3]
        - φext: uniform [0, 2π]
        """
        gamma_ext = np.random.uniform(0.0, 0.3)
        phi_ext = np.random.uniform(0, 2 * np.pi)
        
        # Convert to elliptical components
        gamma_1 = gamma_ext * np.cos(2 * phi_ext)
        gamma_2 = gamma_ext * np.sin(2 * phi_ext)
        
        return al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2)
    
    def sample_source_galaxy(self) -> al.lp.Sersic:
        """
        Sample lensed source galaxy parameters.
        
        Returns
        -------
        Sersic light profile with:
        - r: uniform [0, 0.5]
        - φbk: uniform [0, 2π]
        - z: fixed at 1.0
        - e: uniform [0.4, 1.0] (axis ratio)
        - φ: uniform [0, 2π]
        - n: fixed at 1.5
        - R: uniform [0.25, 0.75]
        """
        # Position in polar coordinates
        r = np.random.uniform(0.0, 0.5)
        phi_bk = np.random.uniform(0, 2 * np.pi)
        
        # Convert to Cartesian
        centre_x = r * np.cos(phi_bk)
        centre_y = r * np.sin(phi_bk)
        
        # Ellipticity
        axis_ratio = np.random.uniform(0.4, 1.0)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Convert ellipticity to components
        ell = 1 - axis_ratio
        ell_1 = ell * np.cos(2 * phi)
        ell_2 = ell * np.sin(2 * phi)
        
        # Effective radius
        effective_radius = np.random.uniform(0.25, 0.75)
        
        # Intensity (brightness)
        intensity = np.random.uniform(0.5, 2.0)
        
        return al.lp.Sersic(
            centre=(centre_x, centre_y),
            ell_comps=(ell_1, ell_2),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.5
        )
    
    def sample_vortex(self) -> al.mp.PointMass:
        """
        Sample vortex parameters (superfluid dark matter).
        
        Returns
        -------
        List of point masses representing vortex line mass distribution:
        - θx: normal [0.0, 0.5]
        - θy: normal [0.0, 0.5]
        - l: uniform [0.5, 2.0]
        - φvort: uniform [0, 2π]
        - mvort: uniform [0.3, 0.5] (% of halo mass)
        """
        # Central position (normal distribution)
        theta_x = np.random.normal(0.0, 0.5)
        theta_y = np.random.normal(0.0, 0.5)
        
        # Vortex parameters
        length = np.random.uniform(0.5, 2.0)
        phi_vort = np.random.uniform(0, 2 * np.pi)
        m_vort_fraction = np.random.uniform(0.3, 0.5)
        
        # Calculate endpoints of vortex line
        dx = (length / 2) * np.cos(phi_vort)
        dy = (length / 2) * np.sin(phi_vort)
        
        # Create line mass distribution with multiple point masses
        n_points = 10
        vortex_masses = []
        
        for i in range(n_points):
            t = (i / (n_points - 1)) - 0.5  # Parameter from -0.5 to 0.5
            x = theta_x + 2 * t * dx
            y = theta_y + 2 * t * dy
            
            # Einstein radius proportional to mass fraction
            einstein_radius = 0.1 * m_vort_fraction / n_points
            
            vortex_masses.append(
                al.mp.PointMass(
                    centre=(x, y),
                    einstein_radius=einstein_radius
                )
            )
        
        return vortex_masses
    
    def sample_subhalos(self) -> list:
        """
        Sample CDM subhalo parameters.
        
        Returns
        -------
        List of NFW profiles representing subhalos:
        - N: Poisson with μ=25
        - r: uniform [0, 2.0]
        - φsh: uniform [0, 2π]
        - msh: power law [1e6, 1e10] M☉
        - βsh: fixed at -1.9
        """
        # Number of subhalos (Poisson distribution)
        n_subhalos = np.random.poisson(25)
        
        subhalos = []
        
        for _ in range(n_subhalos):
            # Position in polar coordinates
            r = np.random.uniform(0.0, 2.0)
            phi_sh = np.random.uniform(0, 2 * np.pi)
            
            # Convert to Cartesian
            x = r * np.cos(phi_sh)
            y = r * np.sin(phi_sh)
            
            # Mass following power law with index -1.9
            # Sample from power law distribution
            beta = -1.9
            m_min, m_max = 1e6, 1e10
            
            # Power law sampling
            u = np.random.uniform(0, 1)
            if beta != -1:
                mass = ((m_max**(beta+1) - m_min**(beta+1)) * u + m_min**(beta+1))**(1/(beta+1))
            else:
                mass = m_min * (m_max/m_min)**u
            
            # Convert mass to Einstein radius (approximate)
            # Einstein radius scales roughly as sqrt(mass)
            einstein_radius = 0.02 * np.sqrt(mass / 1e8)
            
            # Use PointMass for subhalos (simpler and more stable)
            subhalos.append(
                al.mp.PointMass(
                    centre=(x, y),
                    einstein_radius=einstein_radius,
                )
            )
        
        return subhalos
    
    def generate_image(self, 
                      category: str,
                      psf_sigma: float = 0.1,
                      noise_sigma: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a single strong lensing image.
        
        Parameters
        ----------
        category : str
            One of: 'no_substructure', 'cdm_subhalos', 'sfdm_vortices'
        psf_sigma : float
            Sigma of Gaussian PSF in arcseconds
        noise_sigma : float
            Standard deviation of Gaussian noise
            
        Returns
        -------
        image : np.ndarray
            Generated lensing image
        metadata : dict
            Parameters used to generate the image
        """
        metadata = {"category": category}
        
        # Sample lens components (common to all categories)
        dm_halo = self.sample_dm_halo()
        external_shear = self.sample_external_shear()
        
        # Build lens mass profile based on category
        mass_profiles = [dm_halo, external_shear]
        
        if category == 'cdm_subhalos':
            subhalos = self.sample_subhalos()
            mass_profiles.extend(subhalos)
            metadata['n_subhalos'] = len(subhalos)
        elif category == 'sfdm_vortices':
            vortex_masses = self.sample_vortex()
            mass_profiles.extend(vortex_masses)
            metadata['n_vortex_points'] = len(vortex_masses)
        
        # Create lens galaxy
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=dm_halo,
            shear=external_shear
        )
        
        # Add substructure to lens if applicable
        if category == 'cdm_subhalos':
            for i, subhalo in enumerate(subhalos):
                setattr(lens_galaxy, f'subhalo_{i}', subhalo)
        elif category == 'sfdm_vortices':
            for i, vortex_mass in enumerate(vortex_masses):
                setattr(lens_galaxy, f'vortex_{i}', vortex_mass)
        
        # Sample and create source galaxy
        source_profile = self.sample_source_galaxy()
        source_galaxy = al.Galaxy(
            redshift=1.0,
            bulge=source_profile
        )
        
        # Create tracer and generate image
        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
        
        # Generate the lensed image
        image_2d = tracer.image_2d_from(grid=self.grid)
        
        # Convert to numpy array
        if hasattr(image_2d, 'native'):
            image_array = image_2d.native
        else:
            image_array = np.array(image_2d)
        
        # Apply PSF (Gaussian blur) using scipy
        sigma_pixels = psf_sigma / self.pixel_scale
        image_convolved = gaussian_filter(image_array, sigma=sigma_pixels)
        
        # Add noise
        background_sky_level = 0.1
        image_with_sky = image_convolved + background_sky_level
        
        # Add Poisson noise
        noise_map = np.random.normal(0, noise_sigma, size=image_with_sky.shape)
        image = image_with_sky + noise_map
        
        # Add metadata
        metadata.update({
            'lens_redshift': 0.5,
            'source_redshift': 1.0,
            'source_centre': source_profile.centre,
            'source_effective_radius': source_profile.effective_radius,
            'source_sersic_index': source_profile.sersic_index,
            'shear_gamma_1': external_shear.gamma_1,
            'shear_gamma_2': external_shear.gamma_2,
        })
        
        return image, metadata
    
    def generate_dataset(self,
                        n_samples_per_category: int = 1000,
                        save_images: bool = True,
                        show_progress: bool = True) -> Dict[str, list]:
        """
        Generate complete dataset with all three categories.
        
        Parameters
        ----------
        n_samples_per_category : int
            Number of images to generate per category
        save_images : bool
            Whether to save images to disk
        show_progress : bool
            Whether to print progress
            
        Returns
        -------
        dataset : dict
            Dictionary containing images and labels for each category
        """
        categories = ['no_substructure', 'cdm_subhalos', 'sfdm_vortices']
        dataset = {cat: {'images': [], 'metadata': []} for cat in categories}
        
        for cat_idx, category in enumerate(categories):
            if show_progress:
                print(f"\nGenerating {category} images...")
            
            for i in range(n_samples_per_category):
                # Generate image
                image, metadata = self.generate_image(category)
                
                # Store in dataset
                dataset[category]['images'].append(image)
                dataset[category]['metadata'].append(metadata)
                
                # Save to disk
                if save_images:
                    # Save image
                    img_filename = f"{category}_{i:05d}.npy"
                    np.save(self.output_dir / "images" / img_filename, image)
                    
                    # Save metadata
                    meta_filename = f"{category}_{i:05d}.json"
                    with open(self.output_dir / "labels" / meta_filename, 'w') as f:
                        # Convert numpy types to Python types for JSON serialization
                        json_metadata = {}
                        for key, value in metadata.items():
                            if isinstance(value, (np.integer, np.floating)):
                                json_metadata[key] = float(value)
                            elif isinstance(value, tuple):
                                json_metadata[key] = [float(v) for v in value]
                            else:
                                json_metadata[key] = value
                        json.dump(json_metadata, f, indent=2)
                
                if show_progress and (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{n_samples_per_category} images")
        
        if show_progress:
            print(f"\nDataset generation complete!")
            print(f"Total images: {len(categories) * n_samples_per_category}")
            print(f"Output directory: {self.output_dir}")
        
        return dataset
    
    def visualize_samples(self, n_samples: int = 3):
        """
        Generate and visualize sample images from each category.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to show per category
        """
        categories = ['no_substructure', 'cdm_subhalos', 'sfdm_vortices']
        category_labels = ['No Substructure', 'CDM Subhalos', 'SFDM Vortices']
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for col, (category, label) in enumerate(zip(categories, category_labels)):
            for row in range(n_samples):
                image, metadata = self.generate_image(category)
                
                ax = axes[row, col]
                im = ax.imshow(image, cmap='hot', origin='lower')
                ax.set_title(f"{label}\n{category}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_images.png", dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the simulation."""
    # Initialize simulator
    simulator = StrongLensingSimulator(
        output_dir="subhalo_dataset",
        grid_size=100,
        pixel_scale=0.05,
        seed=42
    )
    
    # Generate sample visualizations
    print("Generating sample visualizations...")
    simulator.visualize_samples(n_samples=3)
    
    # Generate full dataset
    print("\nGenerating full dataset...")
    dataset = simulator.generate_dataset(
        n_samples_per_category=1000,  # Adjust as needed
        save_images=True,
        show_progress=True
    )
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
