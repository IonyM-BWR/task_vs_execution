#!/usr/bin/env python3
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm

from bagpy import bagreader
from abc import ABC, abstractmethod

# Local
import evaluation_tools as tools
from evaluation_tools import PATHTYPES

import bwr_modules.bwr_tools as bwr

def get_xy_from_path(path_):
    """
    return x, y: [n], [n] for array of [n,2]
    """
    if path_:
        xy_np = np.array(path_)
        x = np.transpose(xy_np).tolist()[0]
        y = np.transpose(xy_np).tolist()[1]
        return np.array([x,y])

class BasePath(ABC):
    origin_set = False

    def __init__(self, use_pymap_, classname_, path_name_):
        self.classname = classname_
        self._localpath_to_return_enum = None
        self._set_pathname(path_name_)
        self.toggle_pymap_use(use_pymap_, quiet_=True)

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

    def toggle_pymap_use(self, use_pymap_, quiet_=False):
        self._use_pymap = use_pymap_

        if not quiet_:
            s="" if self._use_pymap else "not"
            print(f"{self.get_pathname()} {s} using Pymap")
    
    def set_returntype_forplot(self,path_type_):
        self._localpath_to_return_enum = path_type_

        print(f"PATHTYPE {PATHTYPES.path_types_dict[path_type_]} set for {self.get_pathname()}")
    
class HLCPath(BasePath):
    _localpath_to_return_enum : int

class TaskPath(BasePath):
    def __init__(self, path_to_taskfile__, path_name_, plot_format_, use_pymap_=False):
        self._task_loaded = False
        super().__init__(use_pymap_=use_pymap_, classname_=self.__class__.__name__, path_name_=path_name_)

        self.load_jsontask(path_to_taskfile__)
        if self._task_loaded:
            self._set_pathname(path_name_)
            self._set_plot_format(plot_format_)

            print("\nTaskPath instance created")

    def load_jsontask(self,path_to_task_):
        self._task_loaded = False
        _latlon_path = np.array(bwr.parse_jsontask(path_to_task_))

        if _latlon_path is not None:
            self._latlon_path = _latlon_path
            print(f'json task loaded | {len(self._latlon_path)} waypoints loaded')
            self._task_loaded = True
    
    def get_xy_path(self):
        """return a locally converted lowres from geo_path"""
        if not self.origin_set:
            print(f"ERROR: origin point not defined {self.origin_set}")
            return
        
        if self._use_pymap:
            return [pm.geodetic2enu(lat=coord[0], lon=coord[1], lat0=self._lat_og, lon0=self._lon_og, h=1,h0=1) for coord in self.get_latlon_path()]
        
        return [tools.ll2xy(coord[0], coord[1], self._lat_og, self._lon_og) for coord in self.get_latlon_path()]

    def get_localpath_for_plot(self):
        if self._localpath_to_return_enum == PATHTYPES.LOWRES_PATH:
            _local_path = self.get_xy_path()
            
        elif self._localpath_to_return_enum == PATHTYPES.HIGHRES_PATH:
           raise Exception("PATHTYPES.HIGHRES_PATH not possible for TaskPath")
        
        elif self._localpath_to_return_enum == PATHTYPES.GEO_PATH:
            _local_path = self.get_latlon_path()
        
        return get_xy_from_path(_local_path)
    
    def get_latlon_path(self):
        return self._latlon_path.tolist()

    def get_first_waypoint(self):
        return self._latlon_path[0]
        pass

class RecordedPath(BasePath):
    BAG = 1
    DATALOGGER = 2
    
    def __init__(self, path_to_recordfile_, origin_, path_name_, plot_format_, use_pymap_=False):
        """
        origin_ can be RecordedPath.BAG or RecordedPath.DATALOGGER
        """
        super().__init__(use_pymap_=use_pymap_, classname_=self.__class__.__name__, path_name_=path_name_)

        self._path_loaded = False
        self._geo_path = None
        self._lat_og = None 
        self._lon_og = None

        self.load_recording(path_to_recordfile_, origin_=origin_)
        self._set_plot_format(plot_format_)

        if self._path_loaded:
            print(f"RecordedPath '{path_name_}' created")

    def get_localpath_for_plot(self):
        if self._localpath_to_return_enum == PATHTYPES.LOWRES_PATH:
            _local_path = self.get_xy_path()
            
        elif self._localpath_to_return_enum == PATHTYPES.HIGHRES_PATH:
            _local_path = self._make_highres_from_lowres(self.get_xy_path())
        
        elif self._localpath_to_return_enum == PATHTYPES.GEO_PATH:
            _local_path = self.get_latlon_path()
        
        return get_xy_from_path(_local_path)

    def load_recording(self, path_to_recordfile_, origin_):
        self._path_loaded = False
        tic = time.perf_counter()
        try:
            if origin_==self.BAG:
                self._load_bag_recording(path_to_bag_=path_to_recordfile_)
            elif origin_==self.DATALOGGER:
                self._load_csv_recording(path_to_csv_=path_to_recordfile_)

        except Exception as e:
            print("ERROR loading path")
            print(e)
            
        else:
            toc = time.perf_counter()
            self._path_loaded = False
            print(f"'{path_to_recordfile_}' loaded in {toc - tic:0.4f} seconds to path {self._path_name}")
    
    def _load_bag_recording(self,path_to_bag_):
        b = bagreader(path_to_bag_)
        topic = "/gx5/nav/odom"
        ins_odom_csv = b.message_by_topic(topic) # creates a csv of the topic data
        self.latlon_df = pd.read_csv(ins_odom_csv)
        self.latlon_df.rename(columns={'pose.pose.position.x': 'Latitude', 'pose.pose.position.y': 'Longitude'}, inplace=True)
        self.latlon_df.drop(
            columns=['pose.pose.position.z', 'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w', 'pose.covariance',
            'twist.twist.linear.x', 'twist.twist.linear.y', 'twist.twist.linear.z', 'twist.twist.angular.x', 'twist.twist.angular.y', 'twist.twist.angular.z', 'twist.covariance'],
            axis=1,
            inplace=True)

    def _load_csv_recording(self, path_to_csv_):
        self.latlon_df = pd.read_csv(path_to_csv_)

    def get_latlon_path(self):
        return [[row["Latitude"], row["Longitude"]] for ind, row in self.latlon_df.iterrows()]
    
    def get_xy_path(self):
        if not self.origin_set:
            print(f"ERROR: origin point not defined {self.origin_set}")
            return
        
        if self._use_pymap:
            return [pm.geodetic2enu(lat=coord[0], lon=coord[1], lat0=self._lat_og, lon0=self._lon_og, h=1,h0=1) for coord in self.get_latlon_path()]
        
        return [tools.ll2xy(coord[0], coord[1], self._lat_og, self._lon_og) for coord in self.get_latlon_path()]

    def get_latlon_path_forplot(self):
        lats = np.transpose(self.get_latlon_path()).tolist()[0]
        lons = np.transpose(self.get_latlon_path()).tolist()[1]
        return lats, lons

    def _set_returntype_forplot(self, PATHTYPE_):
        if PATHTYPE_ == PATHTYPES.HIGHRES_PATH:
            super().set_returntype_forplot(PATHTYPES.LOWRES_PATH)
        else:
            super().set_returntype_forplot(PATHTYPE_)

class PathPlotter():
    def __init__(self):
        print("\nPathPlotter instance created")

    def plot_paths(self, list_of_paths, plot_title_):
        ax = plt.axes()
        ax.grid()
        # ax.axis("equal")
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title(plot_title_)

        labels = []
        fuckups = []
        for path in list_of_paths:
            
            try:
                xy = path.get_localpath_for_plot()
                labels.append(path.get_pathname())
                if path._use_pymap:
                    ax.plot(xy[0],xy[1],path.get_plot_format())
                else:
                    ax.plot(xy[0],xy[1],path.get_plot_format())
            except Exception as e:
                x,y = path.get_localpath_for_plot()
                fuckups.append([path.get_pathname(), e])
            finally:
                if fuckups:
                    print(f'this paths could not be printed: {fuckups}')
            ax.legend((labels))
        ax.legend((labels))
        plt.show()
