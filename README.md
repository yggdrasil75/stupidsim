C++ version install instructions:
```
git clone https://github.com/yggdrasil75/stupidsim
cd stupdidsim
python -m venv venv
```

ON WINDOWS:
`venv\scripts\activate`
ON LINUX:
`source venv/bin/activate`

on both:
```
python setup.py build_ext --inplace
python main.py
```
yet another rewrite. still only a couple weeks in so need to finalize the goals before a month to avoid making this too unstable for anyone else to even bother looking at.
Current goals:
Create a reasonably realistic starting point using general world gen techniques to generate a basic world.
This world will be at a point just before civilizations might be considered as starting to form.
at this point you can place an obelisk at points around the world to have various effects such as forcing a sentient species to form in that location or making something ignore natural order at that spot (volcano outside a plate boundary, floating islands, other mystical properties). 
the goal is to run around 1000 years of simulations at 1 year per step to make the world slightly more realistic during the formation of the first societies
next it will slow down and increase detail as the first societies turn into the first civilizations with 1 day being a step and an extra level of subdivisions.
after this step you can then increase detail again on faces with more interesting features (ie: extra detail where a country formed) and continue to progress.
final result will possibly export something that you can import into ue5 or unity somehow and play it like a game. (no guarantees on that part)

how this will be done:
world will be set up as several layers of icospheres with the surface layer being the middle and the upper layers showing atmosphere, while lower shows underground.
using a convolution matrix, the world will process adjacencies using a map of values to vertices in the world object using a multidimensional gradient for inner/outer layers from the surface.
most new features will be implemented with python as a testbed to make sure the idea is sound before being moved to a release version so I can learn c++ while using the language I know (python) to add new features that might not work right.
convolutions will be used to help with this as well.
this project is to learn c++, python, and programming in general. if you know science and want to improve realism of a section, please do make a PR. the end goal might be a world to explore, but the current goal is just to learn how to code.

systems to be implemented in order:
flat world generation and viewing with ability to change layers and see what is going on in the atmosphere and whats happening underground.
plate tectonics (bare minimum for the initialization as a validation step and not as a generation step)
plate edge elevation stuff (mountains, volcanoes, trenches, etc.)
water flow
erosion
multilayered water flow (atmospheric, groundwater)
ground makeup (basic ideas like 3 rock type percentages)
temperature mechanics
later stuff comes later. this will be a lot already.

previous version (probably wont be implemented as these were geographical timeframe ideas):
an attempt to create an excessively realistic fictional world by way of calculating majority of real world factors that impact elevation, rainfall, temperature, etc.


long term goals (some already implemented):
Initial World Parameters (Randomized Seed):

World Size & Shape: Longitude and Latitude (for spherical/cylindrical world), overall dimensions.

Initial Plate Configuration: Number of plates, starting positions, and initial movement vectors (direction and speed). This is crucial for plate tectonics. You can randomize these within plausible ranges.

Initial Crustal Composition: Percentage of continental crust, oceanic crust, and maybe even primordial crust types. This influences mineral distribution later.

Initial Atmospheric Composition: Gases present (Nitrogen, Oxygen, Carbon Dioxide, Water Vapor, Methane, etc.) and their proportions. This sets the stage for climate and life. Think about early Earth vs. modern Earth.

Initial Ocean Coverage: Percentage of surface covered by water.

Initial Temperature Profile: Average global temperature and latitudinal temperature gradients.

B. Plate Tectonics Simulation:

Plate Boundaries: Track the edges of each plate. Types of boundaries (convergent, divergent, transform) will emerge as plates move.

Plate Movement Vectors: Continually update direction and speed of each plate based on internal forces (convection in mantle - you can simplify this with pseudo-random forces or pre-defined patterns).

Crustal Generation/Destruction:

Divergent Boundaries: Create new oceanic crust at mid-ocean ridges (simplified spreading). Volcanism.

Convergent Boundaries: Subduction of oceanic crust under continental or other oceanic crust. Mountain building, volcanic arcs, trenches.

Transform Boundaries: Lateral sliding. Earthquakes.

Fault Lines & Earthquakes: Track stress build-up along plate boundaries and release as earthquakes. Earthquake frequency and magnitude can influence erosion and species development.

Volcanism: Model volcanic eruptions at plate boundaries and hotspots (mantle plumes – can be simplified). Volcanoes create land, release gases into the atmosphere, and deposit minerals.

Mountain Building: Track elevation changes as plates collide. Mountain ranges affect climate, erosion, and create diverse habitats.

C. Erosion Simulation:

Elevation Map: A grid representing the height of the land at each point. This is the primary data structure for terrain.

Rainfall/Precipitation Map: Grid showing rainfall amounts across the world. Influenced by:

Climate Zones: Latitude, wind patterns, ocean currents (simplified).

Orographic Effect: Mountains forcing air upwards and causing precipitation.

Temperature Map: Grid showing temperature across the world. Influenced by:

Latitude: Sunlight intensity.

Altitude: Temperature decreases with elevation.

Ocean Currents: Heat distribution.

Atmospheric Composition (Greenhouse Effect): CO2, methane, etc.

River Systems: Model river formation based on elevation and rainfall. Water flows downhill. Rivers carve landscapes, transport sediment, and are vital resources.

Sedimentation: Track erosion rates based on rainfall, slope, vegetation (once it evolves), and rock type. Sediment is deposited in low-lying areas, river deltas, and oceans.

Weathering: Chemical and physical breakdown of rocks. Influenced by temperature, rainfall, and atmospheric composition.

II. Climate Simulation (The Environment)

A. Global Climate Model (Simplified):

Solar Radiation: Constant input of energy from the sun.

Atmospheric Circulation: Simplified wind patterns (e.g., Hadley cells, Ferrel cells, Polar cells) driven by temperature differences and rotation.

Ocean Currents: Simplified major ocean current patterns distributing heat.

Greenhouse Effect: Calculate the effect of atmospheric gases (CO2, methane, water vapor) on trapping heat.

Albedo: Reflectivity of the surface (ice, snow, vegetation, land, ocean). Changes albedo can create feedback loops (ice-albedo feedback).

Seasonal Cycles: Introduce axial tilt to create seasons and track changes in sunlight intensity and temperature throughout the year.

B. Regional Climate:

Climate Zones: Define climate zones (tropical, temperate, polar, etc.) based on latitude, altitude, and proximity to oceans/mountains.

Microclimates: Local variations in climate due to topography, vegetation, and other factors.

Climate Change: Track long-term changes in global temperature, precipitation patterns, and sea level driven by:

Volcanic Activity: Short-term cooling (ash) and long-term warming (CO2).

Atmospheric Composition Changes: Natural fluctuations and, eventually, biological activity.

Orbital Cycles (Milankovitch Cycles - optional but adds realism): Long-term variations in Earth's orbit and tilt affecting solar radiation.
