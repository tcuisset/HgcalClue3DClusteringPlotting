

# Nominal beam energies
beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]

# Map nominal beam energy -> final beam energy, ie energy of particles in front of detector after synchrotron losses in test beam
synchrotronBeamEnergiesMap:dict[float, float] = {20 : 20, 30 : 30, 50 : 49.99, 80 : 79.93, 100 : 99.83, 120 : 119.65, 150 : 149.14, 200 : 197.32, 250 : 243.61, 300 : 287.18}