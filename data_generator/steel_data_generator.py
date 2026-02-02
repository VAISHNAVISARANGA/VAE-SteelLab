import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import random
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
PROJECT_ROOT = os.path.abspath(".")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMG_DIR = os.path.join(DATA_DIR, "microstructures")
PLOT_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


class SteelComposition:
    """Define steel composition with alloying elements"""
    def __init__(self):
        self.C = random.uniform(0.30, 0.60)  # Carbon %
        self.Mn = random.uniform(0.50, 1.50)  # Manganese %
        self.Si = random.uniform(0.15, 0.40)  # Silicon %
        self.Cr = random.uniform(0.0, 1.50)   # Chromium %
        self.Ni = random.uniform(0.0, 1.00)   # Nickel %
        self.Mo = random.uniform(0.0, 0.50)   # Molybdenum %
        
    def get_composition(self):
        return {
            'C': self.C, 'Mn': self.Mn, 'Si': self.Si,
            'Cr': self.Cr, 'Ni': self.Ni, 'Mo': self.Mo
        }

class CCT_TTT_Calculator:
    """Calculate phase transformations based on CCT/TTT diagrams"""
    
    def __init__(self, composition):
        self.comp = composition
        
    def calculate_Ms_temperature(self):
        """Calculate martensite start temperature (Andrews equation)"""
        C = self.comp['C']
        Mn = self.comp['Mn']
        Cr = self.comp['Cr']
        Ni = self.comp['Ni']
        Mo = self.comp['Mo']
        
        Ms = 539 - 423*C - 30.4*Mn - 17.7*Ni - 12.1*Cr - 7.5*Mo
        return max(200, min(500, Ms))
    
    def calculate_hardenability_factor(self):
        """Calculate DI (Ideal diameter) for hardenability"""
        C = self.comp['C']
        Mn = self.comp['Mn']
        Si = self.comp['Si']
        Cr = self.comp['Cr']
        Ni = self.comp['Ni']
        Mo = self.comp['Mo']
        
        DI_base = 0.5 + 4*C
        DI = DI_base * (1 + 0.64*Mn) * (1 + 0.3*Si) * (1 + 2.33*Cr) * (1 + 0.52*Ni) * (1 + 3*Mo)
        return DI
    
    def get_phase_fractions(self, cooling_rate, quench_temp, aust_temp):
        """Calculate phase fractions based on cooling rate and CCT diagram"""
        Ms = self.calculate_Ms_temperature()
        DI = self.calculate_hardenability_factor()
        
        # Critical cooling rates for different phases
        martensite_cr = 50 * (DI / 5.0)
        bainite_cr_min = 10 * (DI / 5.0)
        bainite_cr_max = martensite_cr
        pearlite_cr = 5 * (DI / 5.0)
        
        martensite = 0
        bainite = 0
        pearlite = 0
        ferrite = 0
        
        # Determine phases based on cooling rate
        if cooling_rate >= martensite_cr:
            # Full martensite transformation
            if quench_temp < Ms:
                # Koistinen-Marburger equation
                martensite = 100 * (1 - np.exp(-0.011 * (Ms - quench_temp)))
            else:
                martensite = 0
            ferrite = 0
            bainite = 0
            pearlite = 100 - martensite
            
        elif bainite_cr_min <= cooling_rate < bainite_cr_max:
            # Bainite + some martensite
            bainite_factor = (cooling_rate - bainite_cr_min) / (bainite_cr_max - bainite_cr_min)
            bainite = 60 * (1 - bainite_factor)
            
            if quench_temp < Ms:
                martensite = 40 * bainite_factor + 20 * (1 - np.exp(-0.011 * (Ms - quench_temp))) * bainite_factor
            else:
                martensite = 0
                
            pearlite = 100 - bainite - martensite
            ferrite = 0
            
        elif pearlite_cr <= cooling_rate < bainite_cr_min:
            # Pearlite + ferrite (slow cooling)
            C_content = self.comp['C']
            if C_content < 0.77:  # Hypoeutectoid
                ferrite = (0.77 - C_content) / 0.77 * 100 * (1 - cooling_rate/bainite_cr_min)
                pearlite = 100 - ferrite
            else:  # Hypereutectoid
                pearlite = 100
                ferrite = 0
            bainite = 0
            martensite = 0
            
        else:
            # Very slow cooling - mostly ferrite + pearlite
            C_content = self.comp['C']
            if C_content < 0.77:
                ferrite = (0.77 - C_content) / 0.77 * 100
                pearlite = 100 - ferrite
            else:
                pearlite = 90
                ferrite = 10
            bainite = 0
            martensite = 0
        
        # Normalize to 100%
        total = martensite + bainite + ferrite + pearlite
        if total > 0:
            martensite = (martensite / total) * 100
            bainite = (bainite / total) * 100
            ferrite = (ferrite / total) * 100
            pearlite = (pearlite / total) * 100
        
        return {
            'martensite': max(0, min(100, martensite)),
            'bainite': max(0, min(100, bainite)),
            'ferrite': max(0, min(100, ferrite)),
            'pearlite': max(0, min(100, pearlite))
        }

class GrainSizeCalculator:
    """Calculate prior austenite grain size"""
    
    @staticmethod
    def calculate_grain_size(aust_temp, aust_time):
        """Calculate grain size based on austenitizing conditions"""
        # Beck equation modified
        base_grain_size = 10
        temp_factor = np.exp((aust_temp - 850) / 150)
        time_factor = np.sqrt(aust_time / 60)
        
        grain_size = base_grain_size * temp_factor * time_factor
        return max(10, min(50, grain_size))

class MechanicalPropertyCalculator:
    """Calculate mechanical properties from microstructure"""
    
    @staticmethod
    def calculate_hardness(phases, grain_size, temp_temp):
        """Calculate Rockwell C hardness"""
        # Base hardness contributions from each phase
        H_martensite = 65
        H_bainite = 45
        H_pearlite = 25
        H_ferrite = 15
        
        # Rule of mixtures
        hardness = (phases['martensite']/100 * H_martensite +
                   phases['bainite']/100 * H_bainite +
                   phases['pearlite']/100 * H_pearlite +
                   phases['ferrite']/100 * H_ferrite)
        
        # Grain size effect (Hall-Petch)
        hardness += 5 * (1/np.sqrt(grain_size))
        
        # Tempering effect
        if temp_temp > 400:
            tempering_reduction = (temp_temp - 400) / 250 * 15
            hardness -= tempering_reduction
        
        return max(20, min(65, hardness))
    
    @staticmethod
    def calculate_yield_strength(hardness, grain_size):
        """Calculate yield strength from hardness"""
        # Empirical correlation: YS (MPa) ≈ 3.45 × HRC
        YS = hardness * 3.45 * 10
        
        # Grain size strengthening
        YS += 150 / np.sqrt(grain_size)
        
        return max(400, min(1600, YS))
    
    @staticmethod
    def calculate_tensile_strength(yield_strength, phases):
        """Calculate ultimate tensile strength"""
        # UTS typically 1.2-1.5 times YS for steels
        ratio = 1.2 + 0.3 * (phases['martensite'] / 100)
        UTS = yield_strength * ratio
        
        return max(600, min(2000, UTS))
    
    @staticmethod
    def calculate_elongation(phases, hardness):
        """Calculate percent elongation (ductility)"""
        # Inverse relationship with hardness
        base_elongation = 30 - (hardness - 20) * 0.4
        
        # Ferrite increases ductility
        ferrite_bonus = phases['ferrite'] / 100 * 5
        
        # Martensite decreases ductility
        martensite_penalty = phases['martensite'] / 100 * 10
        
        elongation = base_elongation + ferrite_bonus - martensite_penalty
        
        return max(5, min(25, elongation))

class MicrostructureImageGenerator:
    """Generate synthetic microstructure images"""
    
    def __init__(self, size=(512, 512)):
        self.size = size
        
    def generate_martensite(self, img, fraction, grain_size):
        """Generate martensitic microstructure (needle-like)"""
        draw = ImageDraw.Draw(img)
        pixels = img.load()
        
        num_needles = int((fraction / 100) * 800 * (50 / grain_size))
        
        for _ in range(num_needles):
            x = random.randint(0, self.size[0]-1)
            y = random.randint(0, self.size[1]-1)
            length = random.randint(int(grain_size*0.5), int(grain_size*1.5))
            angle = random.uniform(0, 180)
            width = random.randint(1, 3)
            
            # Calculate end point
            x2 = int(x + length * np.cos(np.radians(angle)))
            y2 = int(y + length * np.sin(np.radians(angle)))
            
            # Martensite color (light gray to white)
            color = random.randint(200, 240)
            draw.line([(x, y), (x2, y2)], fill=(color, color, color), width=width)
    
    def generate_bainite(self, img, fraction, grain_size):
        """Generate bainitic microstructure (feathery plates)"""
        draw = ImageDraw.Draw(img)
        
        num_plates = int((fraction / 100) * 500 * (50 / grain_size))
        
        for _ in range(num_plates):
            x = random.randint(0, self.size[0]-1)
            y = random.randint(0, self.size[1]-1)
            length = random.randint(int(grain_size*0.8), int(grain_size*2))
            angle = random.uniform(0, 180)
            
            # Multiple parallel lines for feathery appearance
            for offset in range(-2, 3):
                x_off = int(offset * 2 * np.sin(np.radians(angle)))
                y_off = int(offset * 2 * np.cos(np.radians(angle)))
                x1 = x + x_off
                y1 = y + y_off
                x2 = int(x1 + length * np.cos(np.radians(angle)))
                y2 = int(y1 + length * np.sin(np.radians(angle)))
                
                # Bainite color (medium gray)
                color = random.randint(140, 180)
                draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=1)
    
    def generate_pearlite(self, img, fraction, grain_size):
        """Generate pearlitic microstructure (lamellar)"""
        draw = ImageDraw.Draw(img)
        
        num_colonies = int((fraction / 100) * 300 * (50 / grain_size))
        
        for _ in range(num_colonies):
            x = random.randint(0, self.size[0]-1)
            y = random.randint(0, self.size[1]-1)
            size = int(grain_size * random.uniform(1.0, 2.0))
            angle = random.uniform(0, 180)
            
            # Draw lamellar structure
            spacing = 2
            for i in range(0, size, spacing):
                offset_x = int(i * np.cos(np.radians(angle)))
                offset_y = int(i * np.sin(np.radians(angle)))
                
                x1 = x + offset_x
                y1 = y + offset_y
                x2 = int(x1 + size * np.sin(np.radians(angle)))
                y2 = int(y1 - size * np.cos(np.radians(angle)))
                
                # Alternating dark and light lamellae
                if i % 4 == 0:
                    color = random.randint(60, 100)  # Dark (cementite)
                else:
                    color = random.randint(120, 160)  # Light (ferrite)
                
                draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=1)
    
    def generate_ferrite(self, img, fraction, grain_size):
        """Generate ferritic microstructure (equiaxed grains)"""
        draw = ImageDraw.Draw(img)
        
        num_grains = int((fraction / 100) * 200 * (50 / grain_size))
        
        for _ in range(num_grains):
            x = random.randint(0, self.size[0]-1)
            y = random.randint(0, self.size[1]-1)
            size = int(grain_size * random.uniform(0.8, 1.5))
            
            # Draw irregular polygon for grain
            points = []
            num_sides = random.randint(5, 8)
            for j in range(num_sides):
                angle = (j / num_sides) * 360
                r = size * random.uniform(0.7, 1.0)
                px = int(x + r * np.cos(np.radians(angle)))
                py = int(y + r * np.sin(np.radians(angle)))
                points.append((px, py))
            
            # Ferrite color (light)
            color = random.randint(180, 220)
            draw.polygon(points, fill=(color, color, color), outline=(80, 80, 80))
    
    def generate_microstructure(self, phases, grain_size, sample_id):
        """Generate complete microstructure image"""
        # Create base image
        img = Image.new('RGB', self.size, color=(100, 100, 100))
        
        # Add grain boundaries first
        self.add_grain_boundaries(img, grain_size)
        
        # Generate phases in order (background to foreground)
        if phases['ferrite'] > 0:
            self.generate_ferrite(img, phases['ferrite'], grain_size)
        
        if phases['pearlite'] > 0:
            self.generate_pearlite(img, phases['pearlite'], grain_size)
        
        if phases['bainite'] > 0:
            self.generate_bainite(img, phases['bainite'], grain_size)
        
        if phases['martensite'] > 0:
            self.generate_martensite(img, phases['martensite'], grain_size)
        
        # Apply slight blur for more realistic appearance
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Add noise
        img = self.add_noise(img)
        
        return img
    
    def add_grain_boundaries(self, img, grain_size):
        """Add prior austenite grain boundaries"""
        draw = ImageDraw.Draw(img)
        
        # Create grid-like grain structure
        num_grains_x = int(self.size[0] / grain_size)
        num_grains_y = int(self.size[1] / grain_size)
        
        for i in range(num_grains_x + 1):
            for j in range(num_grains_y + 1):
                x = int(i * grain_size + random.uniform(-grain_size*0.2, grain_size*0.2))
                y = int(j * grain_size + random.uniform(-grain_size*0.2, grain_size*0.2))
                
                if i < num_grains_x:
                    x2 = int((i+1) * grain_size + random.uniform(-grain_size*0.2, grain_size*0.2))
                    y2 = y + random.randint(-5, 5)
                    draw.line([(x, y), (x2, y2)], fill=(40, 40, 40), width=2)
                
                if j < num_grains_y:
                    x2 = x + random.randint(-5, 5)
                    y2 = int((j+1) * grain_size + random.uniform(-grain_size*0.2, grain_size*0.2))
                    draw.line([(x, y), (x2, y2)], fill=(40, 40, 40), width=2)
    
    def add_noise(self, img):
        """Add realistic noise to image"""
        pixels = img.load()
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                r, g, b = pixels[i, j]
                noise = random.randint(-10, 10)
                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                b = max(0, min(255, b + noise))
                pixels[i, j] = (r, g, b)
        
        return img

def generate_dataset(num_samples=500):
    """Generate complete dataset with all features"""
    
    print(f"Generating {num_samples} samples...")
     
    data = []
    
    for sample_id in range(num_samples):
        if (sample_id + 1) % 50 == 0:
            print(f"Generated {sample_id + 1}/{num_samples} samples...")
        
        # Generate steel composition
        composition = SteelComposition()
        comp_dict = composition.get_composition()
        
        # Generate process parameters
        aust_temp = random.uniform(800, 950)
        aust_time = random.uniform(30, 120)
        cooling_rate = random.uniform(10, 100)
        quench_temp = random.uniform(20, 200)
        temp_temp = random.uniform(200, 650)
        temp_time = random.uniform(60, 180)
        
        # Calculate microstructure
        cct_calc = CCT_TTT_Calculator(comp_dict)
        phases = cct_calc.get_phase_fractions(cooling_rate, quench_temp, aust_temp)
        
        # Calculate grain size
        grain_size = GrainSizeCalculator.calculate_grain_size(aust_temp, aust_time)
        
        # Calculate mechanical properties
        mech_calc = MechanicalPropertyCalculator()
        hardness = mech_calc.calculate_hardness(phases, grain_size, temp_temp)
        yield_strength = mech_calc.calculate_yield_strength(hardness, grain_size)
        tensile_strength = mech_calc.calculate_tensile_strength(yield_strength, phases)
        elongation = mech_calc.calculate_elongation(phases, hardness)
        
        # Generate microstructure image
        img_gen = MicrostructureImageGenerator()
        img = img_gen.generate_microstructure(phases, grain_size, sample_id)
        img_filename = f'microstructure_{sample_id:04d}.png'
        img.save(os.path.join(IMG_DIR, img_filename))
        
        # Compile data row
        row = {
            'Sample_ID': f'STEEL_{sample_id:04d}',
            'C_percent': round(comp_dict['C'], 3),
            'Mn_percent': round(comp_dict['Mn'], 3),
            'Si_percent': round(comp_dict['Si'], 3),
            'Cr_percent': round(comp_dict['Cr'], 3),
            'Ni_percent': round(comp_dict['Ni'], 3),
            'Mo_percent': round(comp_dict['Mo'], 3),
            'Aust_Temp': round(aust_temp, 1),
            'Aust_Time': round(aust_time, 1),
            'Cooling_Rate': round(cooling_rate, 2),
            'Quench_Temp': round(quench_temp, 1),
            'Temp_Temp': round(temp_temp, 1),
            'Temp_Time': round(temp_time, 1),
            'Martensite_Pct': round(phases['martensite'], 2),
            'Bainite_Pct': round(phases['bainite'], 2),
            'Ferrite_Pct': round(phases['ferrite'], 2),
            'Pearlite_Pct': round(phases['pearlite'], 2),
            'Grain_Size': round(grain_size, 2),
            'Hardness_HRC': round(hardness, 1),
            'Yield_Strength': round(yield_strength, 0),
            'Tensile_Strength': round(tensile_strength, 0),
            'Elongation': round(elongation, 1),
            'Microstructure_Image': img_filename
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns according to schema
    column_order = [
        'Sample_ID',
        'C_percent', 'Mn_percent', 'Si_percent', 'Cr_percent', 'Ni_percent', 'Mo_percent',
        'Aust_Temp', 'Aust_Time', 'Cooling_Rate', 'Quench_Temp', 'Temp_Temp', 'Temp_Time',
        'Martensite_Pct', 'Bainite_Pct', 'Ferrite_Pct', 'Pearlite_Pct', 'Grain_Size',
        'Hardness_HRC', 'Yield_Strength', 'Tensile_Strength', 'Elongation',
        'Microstructure_Image'
    ]
    
    df = df[column_order]
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(df)}")
    print(f"Microstructure images saved")
    
    return df

# Generate the dataset
df = generate_dataset(num_samples=500)

# Save to CSV
csv_path = os.path.join(DATA_DIR, "steel_heat_treatment_data.csv")
df.to_csv(csv_path, index=False)

print(f"\nCSV saved to: {csv_path}")
print(f"\nDataset Statistics:")
print(df.describe())

# Create a summary visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Steel Heat Treatment Dataset - Distribution Summary', fontsize=16)

# Plot distributions
axes[0, 0].hist(df['Hardness_HRC'], bins=30, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Hardness Distribution')
axes[0, 0].set_xlabel('HRC')

axes[0, 1].hist(df['Martensite_Pct'], bins=30, color='darkred', edgecolor='black')
axes[0, 1].set_title('Martensite %')
axes[0, 1].set_xlabel('Percentage')

axes[0, 2].hist(df['Grain_Size'], bins=30, color='green', edgecolor='black')
axes[0, 2].set_title('Grain Size Distribution')
axes[0, 2].set_xlabel('μm')

axes[1, 0].scatter(df['Cooling_Rate'], df['Hardness_HRC'], alpha=0.5, color='purple')
axes[1, 0].set_title('Cooling Rate vs Hardness')
axes[1, 0].set_xlabel('Cooling Rate (°C/s)')
axes[1, 0].set_ylabel('Hardness (HRC)')

axes[1, 1].scatter(df['Tensile_Strength'], df['Elongation'], alpha=0.5, color='orange')
axes[1, 1].set_title('Strength vs Ductility Trade-off')
axes[1, 1].set_xlabel('Tensile Strength (MPa)')
axes[1, 1].set_ylabel('Elongation (%)')

# Phase composition pie chart (average)
phase_avg = [df['Martensite_Pct'].mean(), df['Bainite_Pct'].mean(), 
             df['Ferrite_Pct'].mean(), df['Pearlite_Pct'].mean()]
axes[1, 2].pie(phase_avg, labels=['Martensite', 'Bainite', 'Ferrite', 'Pearlite'],
               autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold'])
axes[1, 2].set_title('Average Phase Composition')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "dataset_summary.png"), dpi=300, bbox_inches='tight')
print(f"\nSummary visualization saved.")

# Display sample microstructures
sample_indices = [0, 100, 200, 300, 400]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Sample Microstructures', fontsize=16)

for idx, sample_idx in enumerate(sample_indices):
    img_path = os.path.join(IMG_DIR, df.iloc[sample_idx]["Microstructure_Image"])
    img = Image.open(img_path)
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f'Sample {sample_idx}\n'
                       f'M:{df.iloc[sample_idx]["Martensite_Pct"]:.1f}% '
                       f'B:{df.iloc[sample_idx]["Bainite_Pct"]:.1f}%\n'
                       f'HRC:{df.iloc[sample_idx]["Hardness_HRC"]:.1f}')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,"sample_microstructures.png"), dpi=300, bbox_inches='tight')
print(f"Sample microstructures visualization saved.")

print("\n✓ Generation Complete!")
print(f"✓ 500 samples generated")
print(f"✓ 500 microstructure images created")
print(f"✓ CSV with all data saved")
print(f"✓ Summary visualizations created")