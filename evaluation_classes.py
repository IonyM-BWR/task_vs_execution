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
    if path_:
        xy_np = np.array(path_)
        x = np.transpose(xy_np).tolist()[0]
        y = np.transpose(xy_np).tolist()[1]
        return x,y

class BasePath(ABC):
    origin_set = False

    @abstractmethod
    def get_localpath_for_plot(self):
        pass

    def _set_plot_format(self, format_):
        self._plot_format = format_
    
    def get_plot_format(self):
        return self._plot_format
    
    def _set_pathname(self,path_name_):
        self._path_name = path_name_

    def get_pathname(self):
        return self._path_name
    
    def define_origin_coordinates(self, lat_, lon_):
        self._lat_og = lat_
        self._lon_og = lon_
        self.origin_set = True
        print(f'Origin coordinates set to {lat_, lon_}')

class HLCPath(BasePath):
    _localpath_to_return_enum : int

    def __init__(self,path_to_csv_,path_name_, plot_format_):
        self._localpath_to_return_enum = None
        """
        Object to process the task geo_path.csv produced by the HLC from the autonomous tractor project
        """
        self._set_path(_path_to_csv=path_to_csv_)
        self._set_pathname(path_name_)
        self._set_plot_format(plot_format_)
        print(f"HLCPath '{path_name_}' created")
    
    def _set_path(self, _path_to_csv):
        with open(_path_to_csv, 'r') as f:
            # self._geo_path = np.array(list(csv.reader(f, delimiter=",")))
            self._geo_path = pd.read_csv(_path_to_csv).values

            print(f"path set with {self._geo_path.shape} waypoints")

    def _set_pathname(self,name_):
        self._path_name = name_
        if not self._localpath_to_return_enum:
            print('PATHTYPE not defined yet')

    def get_pathname(self):
        return f"{super().get_pathname()}_{PATHTYPES.path_types_dict.get(self._localpath_to_return_enum)}"

    def load_path_fromcsv(self,path_to_csv_):
        self._set_path(path_to_csv_)
        print(f"Path loaded with {len(self._geo_path)} waypoints")
    
    def get_localpath_for_plot(self):
        if self._localpath_to_return_enum == PATHTYPES.LOWRES_PATH:
            _local_path = self._get_lowres_path()
            
        elif self._localpath_to_return_enum == PATHTYPES.HIGHRES_PATH:
            _local_path = self._make_highres_from_lowres(self._get_lowres_path())
        
        return get_xy_from_path(_local_path)

    def _get_lowres_path(self):
        """return a locally converted lowres from geo_path"""
        if not self.origin_set:
            print(f"ERROR: origin point not defined {self.origin_set}")
            return

        return [tools.ll2xy(coord[0], coord[1], self._lat_og, self._lon_og) for coord in self.get_latlon_path()]

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

            # if intermediate_pts > 4: print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, L:{L}, int_pts:{intermediate_pts}")
            # if idx % 50 == 0: print(f"pts_np:{pts_np}")
            
            for point in pts_np:
                highres_path.append(point.tolist())
        
        return highres_path

    def get_latlon_path(self):
        return self._geo_path

    def set_returntype_forplot(self,path_type_):
        self._localpath_to_return_enum = path_type_
        self._set_plot_format(f"{self._plot_format}" if path_type_ == PATHTYPES.LOWRES_PATH else f"-{self._plot_format}")

class TaskPath(BasePath):
    def __init__(self, path_to_taskfile__, path_name_, plot_format_):
        self.load_jsontask(path_to_taskfile__)
        self._set_pathname(path_name_)
        self._set_plot_format(plot_format_)

        print("\nTaskPath instance created")

    def load_jsontask(self,path_to_task_):
        self._latlon_path = np.array(tools.new_parse_jsontask(path_to_task_))
        print(f'json task loaded | {len(self._latlon_path)} waypoints loaded')
    
    def get_xy_path(self):
        if not self.origin_set:
            print(f"ERROR: origin point not defined {self.origin_set}")
            return

        return [tools.ll2xy(coord[0], coord[1], self._lat_og, self._lon_og) for coord in self._latlon_path]

    def get_localpath_for_plot(self):
        return get_xy_from_path(self.get_xy_path())
    
    def get_latlon_path(self):
        return self._latlon_path

    def get_first_waypoint(self):
        return self._latlon_path[0]
        pass

class RecordedPath(BasePath):
    csv_df : pd.DataFrame
    latlon_path : list
    xy_path : list
    
    def __init__(self, path_to_recordfile_, path_name_, plot_format_):
        self._geo_path = None
        self._lat_og = None 
        self._lon_og = None

        self._set_pathname(path_name_)
        self.load_recording(path_to_recordfile_)
        self._set_plot_format(plot_format_)

        print(f"RecordedPath '{path_name_}' created")

    def get_localpath_for_plot(self):
        return get_xy_from_path(self.get_xy_path())

    def load_recording(self, path_to_csvfile):
        tic = time.perf_counter()
        try:
            self.csv_df = pd.read_csv(path_to_csvfile)
        except Exception as e:
            print("ERROR loading path csv")
            print(e)
        else:
            self.csv_df[['Latitude','Longitude']].apply(lambda x: x/100) #possibly the coolest thing ive ever seen
            toc = time.perf_counter()
            print(f"'{path_to_csvfile}' loaded in {toc - tic:0.4f} seconds to path {self._path_name}")    
    
    def get_latlon_path(self):
        return [[row["Latitude"], row["Longitude"]] for ind, row in self.csv_df.iterrows()]
    
    def get_xy_path(self):
        if not self.origin_set:
            print(f"ERROR: origin point not defined {self.origin_set}")
            return

        return [tools.ll2xy(coord[0], coord[1], self._lat_og, self._lon_og) for coord in self.get_latlon_path()]

    def get_latlon_path_forplot(self):
        lats = np.transpose(self.get_latlon_path()).tolist()[0]
        lons = np.transpose(self.get_latlon_path()).tolist()[1]
        return lats, lons

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
        fuckups = []
        for path in list_of_paths:
            x,y = path.get_localpath_for_plot()
            labels.append(path.get_pathname())
            ax.plot(x,y,path.get_plot_format())
            # try:
            #     x,y = path.get_localpath_for_plot()
            # except Exception as e:
            #     x,y = path.get_localpath_for_plot()
            #     fuckups.append([path.get_pathname(), e])
            # else:
            #     ax.plot(x,y,path.get_plot_format())
            # finally:
            #     if fuckups:
            #         print(f'this paths could not be printed: {fuckups}')
            # ax.legend((labels))
        ax.legend((labels))
