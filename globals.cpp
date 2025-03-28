#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <string>
#include "json.hpp"

using json = nlohmann::json;

json config;


const double PLANET_RADIUS_KM = 6371.0;
const double ORBITAL_DISTANCE_AU = 1.0;
const double PLANET_MASS_EARTH = 1.0;
const double GAS_CONSTANT = 287.05;
constexpr double SPEED_OF_LIGHT 			= 299792458.0;			// m/s
constexpr double GRAVITATIONAL_CONSTANT		= 6.67430e-11;			// N·m²/kg²
constexpr double PLANCK_CONSTANT			= 6.62607015e-34;		// J·s
constexpr double VACUUM_PERMITTIVITY		= 8.8541878128e-12;		// F/m
constexpr double ELEMENTARY_CHARGE			= 1.602176634e-19;		// C
constexpr double BOLTZMANN_CONSTANT			= 1.380649e-23;			// J/K

constexpr double STEFAN_BOLTZMANN			= 5.670374419e-8;		// W/m²K⁴  
constexpr double MASS_LUMINOSITY_ALPHA		= 3.5;					// L ∝ M^α (approx for M > 0.43 Msun)  
constexpr double MASS_RADIUS_BETA			= 0.8;					// R ∝ M^β (approx for Sun-like stars)  

constexpr double DARK_MATTER_DENSITY		= 0.4;					// GeV/cm³
constexpr double INTERSTELLAR_DENSITY		= 1.0;					// atoms/cm³
constexpr double MAGNETIC_FIELD				= 10.0;					// microgauss (μG)

// Interstellar medium (scalable)  
constexpr double ISM_IONIZATION_RATE		= 1.0e-17;				// s⁻¹ per H₂ (cosmic ray baseline)  
constexpr double DUST_OPACITY				= 10.0;					// cm²/g (ISM dust absorption)  

// Planetary formation (universal)  
constexpr double ICE_LINE_TEMPERATURE		= 150.0;				// K (water snow line at ~2.7 AU for Sun)  
constexpr double ROCK_VAPORIZATION_TEMP		= 1400.0;				// K (silicate condensation)  

// Space weather (scales with stellar activity)  
constexpr double COSMIC_RAY_FLUX_GALACTIC	= 2.0;					// particles/cm²/s (≥1 GeV, anywhere in MW)  
constexpr double MAGNETIC_STRENGTH_STAR		= 1.0e-3;				// T (1 kG, typical active star) 

std::map<std::string, double> ATMOSPHERIC_COMPOSITION = {
    {"N2", 0.7808},  // Nitrogen
    {"O2", 0.2095},  // Oxygen
    {"Ar", 0.0093},  // Argon
    {"CO2", 0.0004}, // Carbon dioxide
    {"H2O", 0.01},   // Water vapor (variable)
    {"CH4", 1.8e-6}, // Methane
    {"O3", 7.0e-6}   // Ozone (variable)
};
std::map<std::string, double> MOLECULAR_WEIGHTS = {
	{"N2", 0.0280134},
    {"O2", 0.0319988},
    {"Ar", 0.039948},
    {"O2", 0.0440095},
    {"H2O", 0.01801528},
    {"CH4", 0.0160425},
    {"O3", 0.0479982}
};