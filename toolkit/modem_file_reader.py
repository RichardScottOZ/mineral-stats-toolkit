# -*- coding: utf-8 -*-
"""
   Copyright 2022 Commonwealth of Australia (Geoscience Australia)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Custom ModEM file readers to replace mtpy dependency.
This module provides functions to read ModEM .rho (resistivity model) files
and .dat (data/station) files without requiring the deprecated mtpy library.
"""
import numpy as np


class ModelFileReader:
    """
    Reader for ModEM resistivity model (.rho) files.
    
    This class replaces the functionality of mtpy.modeling.modem.Model
    for reading ModEM .rho files without the mtpy dependency.
    """
    
    def __init__(self):
        self.grid_east = None
        self.grid_north = None
        self.grid_z = None
        self.res_model = None
        self.nodes_east = None
        self.nodes_north = None
        self.nodes_z = None
        
    def read_model_file(self, model_fn):
        """
        Read a ModEM resistivity model file (.rho format).
        
        Parameters
        ----------
        model_fn : str
            Full path to the model file
            
        Returns
        -------
        None
            Sets the following attributes:
            - grid_east: array of east coordinates of grid edges
            - grid_north: array of north coordinates of grid edges  
            - grid_z: array of depth coordinates of grid edges
            - res_model: 3D array of resistivity values [ny, nx, nz]
        """
        with open(model_fn, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines (starting with #)
        line_idx = 0
        while line_idx < len(lines) and lines[line_idx].strip().startswith('#'):
            line_idx += 1
        
        # Parse header line
        # Format: nx ny nz type [optional comments]
        header = lines[line_idx].strip().split()
        nx, ny, nz = int(header[0]), int(header[1]), int(header[2])
        
        # Determine if values are in log10 or linear
        if len(header) > 3:
            value_type = header[3].upper()
        else:
            value_type = 'LOGE'  # default
        
        # Read cell sizes for each direction
        line_idx += 1
        
        # Read north cell sizes (y direction)
        nodes_north = []
        while len(nodes_north) < ny:
            line_data = lines[line_idx].strip().split()
            if line_data:  # Skip empty lines
                nodes_north.extend([float(val) for val in line_data])
            line_idx += 1
        nodes_north = np.array(nodes_north[:ny])
        
        # Read east cell sizes (x direction)
        nodes_east = []
        while len(nodes_east) < nx:
            line_data = lines[line_idx].strip().split()
            if line_data:  # Skip empty lines
                nodes_east.extend([float(val) for val in line_data])
            line_idx += 1
        nodes_east = np.array(nodes_east[:nx])
        
        # Read depth cell sizes (z direction)
        nodes_z = []
        while len(nodes_z) < nz:
            line_data = lines[line_idx].strip().split()
            if line_data:  # Skip empty lines
                nodes_z.extend([float(val) for val in line_data])
            line_idx += 1
        nodes_z = np.array(nodes_z[:nz])
        
        # Store node sizes
        self.nodes_north = nodes_north
        self.nodes_east = nodes_east
        self.nodes_z = nodes_z
        
        # Calculate grid coordinates (cumulative sum of cell sizes)
        # Grid coordinates are at cell edges, starting from 0
        self.grid_north = np.concatenate([[0], np.cumsum(nodes_north)])
        self.grid_east = np.concatenate([[0], np.cumsum(nodes_east)])
        self.grid_z = np.concatenate([[0], np.cumsum(nodes_z)])
        
        # Read resistivity values
        # Values are organized by z-slice, with each slice containing ny*nx values
        res_values = []
        while line_idx < len(lines):
            line_data = lines[line_idx].strip().split()
            if line_data:  # Skip empty lines
                res_values.extend([float(val) for val in line_data])
            line_idx += 1
        
        # Reshape resistivity array to [nz, ny, nx]
        # Data is ordered as: depth -> row (north) -> column (east)
        # Then transpose to [ny, nx, nz] to match mtpy convention
        res_array = np.array(res_values[:nz * ny * nx])
        res_model_3d = res_array.reshape(nz, ny, nx)
        self.res_model = res_model_3d.transpose(1, 2, 0)  # [ny, nx, nz]
        
        # Convert from log10 to linear if necessary
        if value_type in ['LOGE', 'LOG10']:
            self.res_model = 10 ** self.res_model


class DataFileReader:
    """
    Reader for ModEM data files (.dat format) with station locations.
    
    This class replaces the functionality of mtpy.modeling.modem.Data
    for reading ModEM .dat files without the mtpy dependency.
    """
    
    def __init__(self, model_epsg=None):
        self.model_epsg = model_epsg
        self.station_locations = StationLocations()
        self.center_point = {'east': 0.0, 'north': 0.0, 'lon': 0.0, 'lat': 0.0}
        
    def read_data_file(self, data_fn):
        """
        Read a ModEM data file (.dat format) to extract station locations.
        
        Parameters
        ----------
        data_fn : str
            Full path to the data file
            
        Returns
        -------
        None
            Sets the station_locations attribute with lon/lat arrays
        """
        with open(data_fn, 'r') as f:
            lines = f.readlines()
        
        # Parse header to find station information
        # ModEM .dat files have a specific format with header lines
        # followed by data blocks
        
        stations = {}
        data_started = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Header section typically contains format info
            # Data section starts after header
            if line.startswith('>'):
                data_started = True
                continue
            
            if data_started:
                # Parse data lines which contain station info
                # Format varies but typically: period site_name east north data...
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Try to extract station coordinates
                        # Typically: period(0) site(1) east(2) north(3) ...
                        site = parts[1]
                        east = float(parts[2])
                        north = float(parts[3])
                        if site not in stations:
                            stations[site] = (east, north)
                    except (ValueError, IndexError):
                        # If parsing fails, skip this line
                        continue
        
        # Extract unique station locations
        if stations:
            station_coords = list(stations.values())
            rel_east = np.array([coord[0] for coord in station_coords])
            rel_north = np.array([coord[1] for coord in station_coords])
            
            # Store as relative coordinates
            self.station_locations.rel_east = rel_east
            self.station_locations.rel_north = rel_north
            
            # Calculate center point
            self.center_point['east'] = np.mean(rel_east)
            self.center_point['north'] = np.mean(rel_north)
            
            # Note: lon/lat will need to be calculated using projection
            # This is typically done in the ModEM class using epsg_project


class StationLocations:
    """
    Container for station location data.
    Mimics the structure of mtpy station_locations.
    """
    
    def __init__(self):
        self.lon = None
        self.lat = None
        self.rel_east = None
        self.rel_north = None
