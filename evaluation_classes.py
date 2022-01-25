#!/usr/bin/env python3
import time, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

# Local
import evaluation_tools as tools
from evaluation_tools import PATHTYPES

def get_xy_from_path(path_):
    """
    return x, y: [n], [n] for array of [n,2]
    """
    xy_np = np.array(path_)
    x = np.transpose(xy_np).tolist()[0]
    y = np.transpose(xy_np).tolist()[1]
    return x,y

class BasePath(ABC):
    @abstractmethod
    def get_localpath_for_plot(self):
        pass

    @abstractmethod
    def _set_plot_format(self,format_):
        pass
    
    @abstractmethod
    def get_plot_format(self):
        pass

    @abstractmethod
    def get_pathname(self):
        pass
    @abstractmethod
    def _set_pathname(self):
        pass


class HLCPath(BasePath):
    def __init__(self,path_to_csv_,name_):
        """
        Object to process the task geo_path.csv produced by the HLC from the autonomous tractor project
        """
        self._geo_path = None
        self._lat_og = None 
        self._lon_og = None

        self._set_path(_path_to_csv=path_to_csv_)
        self._set_pathname(name_)
        print(f"HLCPath '{name_}' created")
    
    def _set_path(self, _path_to_csv):
        with open(_path_to_csv, 'r') as f:
            # self._geo_path = np.array(list(csv.reader(f, delimiter=",")))
            self._geo_path = pd.read_csv(_path_to_csv).values

            print(f"path set with {self._geo_path.shape} waypoints")

    def _set_pathname(self,name_):
        self._name = name_

    def load_path_fromcsv(self,path_to_csv_):
        self._set_path(path_to_csv_)
        print(f"Path loaded with {len(self._geo_path)} waypoints")

    def define_origin_coordinate(self, lat_, lon_):
        self._lat_og = lat_ 
        self._lon_og = lon_
    
    def get_localpath_for_plot(self):
        if self._localpath_to_return == PATHTYPES.LOWRES_PATH:
            _local_path = self._get_lowres_path()
            
        elif self._localpath_to_return == PATHTYPES.HIGHRES_PATH:
            _local_path = self._make_highres_from_lowres(self._get_lowres_path())
        
        return get_xy_from_path(_local_path)

    def _get_lowres_path(self):
        """return a locally converted lowres from geo_path"""

        lowres_path = []
        for coord in self._geo_path:
            lowres_path.append(tools.ll2xy(coord[0], coord[1], self._lat_og, self._lon_og))
            
        return lowres_path

    def _make_highres_from_lowres(self, lowres_path_):
        # parameters for making highres
        points_per_meter = 4

        highres_path = []
        for idx in range(len(lowres_path_)-1):
            x1, y1 = lowres_path_[idx][0], lowres_path_[idx][1]
            x2, y2 = lowres_path_[idx+1][0], lowres_path_[idx][1]
            L = np.hypot(np.abs(x2-x1), np.abs(y2-y1))

            intermediate_pts = round(points_per_meter*L)
            pts_np = np.linspace([x1, y1],[x1, y2], num=intermediate_pts+2,endpoint=False)

            for point in pts_np:
                highres_path.append(point.tolist())
        
        return highres_path

    def get_geopath(self):
        return self._geo_path

    def get_pathname(self):
        return self._name

    def set_returntype_forplot(self,path_type_):
        self._localpath_to_return = path_type_
        self._set_plot_format("x" if path_type_ == PATHTYPES.LOWRES_PATH else '-o')

    def _set_plot_format(self, format_):
        self._plot_format = format_
    
    def get_plot_format(self):
        return self._plot_format

class TaskPath():
    latlon_path : list
    reference_point : list
    xy_lowres_path : list
    xy_highres_path : list

    def __init__(self, taskpath=None) -> None:
        self.task_name = ""
        self.latlon_path = None
        self.reference_point = None
        self.xy_lowres_path = None
        self.xy_highres_path = None
        self._set_plot_format('o')
        print("\nTaskPath instance created")

    def load_jsontask(self,taskname):
        self.task_name = taskname
        self.latlon_path = np.array(tools.new_parse_jsontask(self.task_name))
        self._define_origin_coords()
        print(f'json task loaded | {len(self.latlon_path)} waypoints loaded')
        
        self._make_xy_paths()

    def _define_origin_coords(self):
        self.reference_point = self.latlon_path[0]
        print('Origin coordinates defined')
    
    def _make_xy_paths(self):
        self.xy_lowres_path = []
        num = 0
        for i, coord in enumerate(self.latlon_path):
            self.xy_lowres_path.append(tools.ll2xy(coord[0], coord[1], self.reference_point[0], self.reference_point[1]))
            num += i

    def get_pathname(self):
        return self.task_name
    
    def get_lowres_path(self):
        return np.array(self.xy_lowres_path)
    
    def get_localpath_for_plot(self):
        x = np.transpose(self.get_lowres_path()).tolist()[0]
        y = np.transpose(self.get_lowres_path()).tolist()[1]
        return x,y
    
    def get_latlon_path_forplot(self):
        lats = np.transpose(self.get_latlon_path()).tolist()[0]
        lons = np.transpose(self.get_latlon_path()).tolist()[1]
        return lats, lons
    
    def get_latlon_path(self):
        return self.latlon_path
    
    def get_reference_coords(self):
        return self.latlon_path[0]

    def _set_plot_format(self, format_):
        self._plot_format = format_
    
    def get_plot_format(self):
        return self._plot_format

class RecordedPath():
    csv_df : pd.DataFrame()
    latlon_path : list
    xy_path : list
    
    def __init__(self,recording_filename_ = None):
        print("\nRecordedPath instance created")
        if recording_filename_:
            self.filename = recording_filename_
            self.load_recording(recording_filename_)
            self._make_latlonpath()
        
        self.origin_set = True
  
    def load_recording(self, path_to_csvfile):
        tic = time.perf_counter()
        try:
            self.csv_df = pd.read_csv(path_to_csvfile)
        except Exception as e:
            print(e)
        else:
            self.filename = path_to_csvfile
            self.csv_df[['Latitude','Longitude']].apply(lambda x: x/100) #possibly the coolest thing ive ever seen
            toc = time.perf_counter()
            print(f"'{path_to_csvfile}' loaded in {toc - tic:0.4f} seconds")    
    
    def _make_latlonpath(self):
        tic = time.perf_counter()
        self.latlon_path = [[row["Latitude"], row["Longitude"]] for ind, row in self.csv_df.iterrows()]
        toc = time.perf_counter()

        print(f"latlon_path created with {len(self.latlon_path)} coordinates in {toc - tic:0.4f} seconds")
    
    def define_origin_coords(self, origin_coords):
        self.ref_point = origin_coords
        self.origin_set = True
        print(f'Origin coordinates set to {self.ref_point}')
        self._make_xy_path()
        
    def _make_xy_path(self):
        tic = time.perf_counter()
        self.xy_path = []
        if not self.origin_set:
            print(f"ERROR: origin point not defined {self.origin_set}")
            return

        for coord in self.latlon_path:
            self.xy_path.append(tools.ll2xy(coord[0], coord[1], self.ref_point[0], self.ref_point[1]))
        toc = time.perf_counter()

        print(f"xy_lowres_path created with {len(self.xy_path)} waypoints in {toc - tic:0.4f} seconds")
    
    def get_filename(self):
        return self.filename
    
    def get_xypath(self):
        if not self.origin_set: print("Set origin coordinates before requesting xy_path")
        return np.array(self.xy_path)
    
    def get_latlon_path_forplot(self):
        lats = np.transpose(self.get_latlon_path()).tolist()[0]
        lons = np.transpose(self.get_latlon_path()).tolist()[1]
        return lats, lons

    def get_latlon_path(self):
        return np.array(self.latlon_path)
    
class PathPlotter():
    def __init__(self):
        print("\nPathPlotter instance created")

    def plot_paths(self, list_of_paths, plot_title_):
        ax = plt.axes()
        ax.grid()
        ax.axis("equal")
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title(plot_title_)

        labels = []
        for path in list_of_paths:
            x,y = path.get_localpath_for_plot()
            labels.append(path.get_pathname())
            ax.plot(x,y,path.get_plot_format())
        ax.legend((labels))
