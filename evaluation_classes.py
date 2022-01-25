#!/usr/bin/env python3
import time, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

# Local
import evaluation_tools as tools
from evaluation_tools import PATHTYPES

class BasePath(ABC):
    @abstractmethod
    def load_path(self):
        pass

class HLCPath():
    def __init__(self,path_type_=None):
        self.set_pathtype(path_type_)

        print("HLC created")
    
    def _set_path(self, path_to_csv):
        
        with open(path_to_csv, 'r') as f:
            self._path = np.array(list(csv.reader(f, delimiter=",")))

            print(f"path set with {self._path.shape} waypoints")

        return True

    def set_pathtype(self, path_type_:int):
        if path_type_ not in PATHTYPES.path_types_dict.keys() and path_type_ is not None:
            print(f"PATH_TYPE {path_type_} not recognized")
            return False

        self._path_type = path_type_
        self._path_type_str = PATHTYPES.path_types_dict.get(path_type_)
        
        if path_type_: print(f"path type set as {self._path_type_str}")        
        return True

    def load_path_fromcsv(self,path_to_csv_, path_type_):
        success = self._set_path(path_to_csv_)
        success &= self.set_pathtype(path_type_)
        print(f"{self._path_type_str} path loaded with {len(self._path)} waypoints")

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

    def get_filename(self):
        return self.task_name
    
    def get_lowres_path(self):
        return np.array(self.xy_lowres_path)
    
    def get_lowres_path_forplot(self):
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
    selected_task_path : TaskPath
    task_paths : dict # task_path_.get_filename() : task_path_
    selected_recorded_path : RecordedPath
    recorded_paths : dict # [RecordedPath.get_filename(), RecordedPath]
        
    def __init__(self, task_path_ = None, recorded_path_ = None):
        self.recorded_paths = dict()
        self.task_paths = dict()
        
        if task_path_:
            self.selected_task_path = task_path_
            self.add_taskpath(task_path_)
        if recorded_path_:
            self.recorded_path = recorded_path_
            self.add_recordedpath(recorded_path_)
            
        # Plotting variables
            
        print("\nPathPlotter instance created")
        
    def add_taskpath(self, task_path_: TaskPath):
        if task_path_.get_filename() in self.task_paths.keys():
            print(f"ERRROR: {task_path_.get_filename()} already loaded in the PathPlotter")
            self.list_all_taskpaths()
            return

        self.task_paths[task_path_.get_filename()] = task_path_
        print(f"Task path '{task_path_.get_filename()}' added to PathPlotter")

        self.select_task_from_taskpaths(task_path_.get_filename())
    
    def select_task_from_taskpaths(self, task_name:str):
        if task_name not in self.task_paths.keys():
            print(f"ERROR: '{task_name}' not in task_paths")
            self.list_all_taskpaths()
            return
        
        self.set_task_as_selected(self.task_paths.get(task_name))
        print(f"selected task path: {self.selected_task_path.get_filename()}")
        
    def set_task_as_selected(self,task_path:TaskPath):
        self.selected_task_path = task_path

    def add_recordedpath(self, recorded_path_: RecordedPath):
        if recorded_path_.get_filename() in self.recorded_paths.keys():
            print(f"ERRROR: '{recorded_path_.get_filename()}' already loaded in the PathPlotter")
            self.list_all_recordpaths()
            return
        
        self.recorded_paths[recorded_path_.get_filename()] = recorded_path_
        print(f"Recorded path '{recorded_path_.get_filename()}' added to PathPlotter")
        
        self.select_record_from_recordedpaths(recorded_path_.get_filename())
    
    def select_record_from_recordedpaths(self, record_name:str):
        if record_name not in self.recorded_paths.keys():
            print(f"ERROR: '{record_name}' not in task_paths")
            self.list_all_recordpaths()
            return
        
        self.set_record_as_selected(self.recorded_paths.get(record_name))
        print(f"selected record path: {self.selected_record_path.get_filename()}")

    def set_record_as_selected(self,recorded_path:RecordedPath):
        self.selected_record_path = recorded_path

    def plot_selected_task(self):
        if self.selected_task_path:
            x,y = self.selected_task_path.get_lowres_path_forplot()
            ax = plt.axes()
            # plt.figure()
            ax.grid()
            ax.axis("equal")
            ax.plot(x,y, "xg")
            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')
            plt.title(self.selected_task_path.get_filename())
            plt.legend(("hello"))

        else:
            print('ERROR: No task path loaded')
    
    def plot_all_tasks(self):
        ax = plt.axes()
        ax.grid()
        ax.axis("equal")
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title(self.selected_task_path.get_filename())

        labels = []

        for task_name, task_path in self.task_paths.items():
            x,y = task_path.get_lowres_path_forplot()
            labels.append(task_name)
            ax.plot(x,y)
        ax.legend((labels))

    def plot_recorded(self):
        if self.selected_recorded_path:
            x,y = self.selected_recorded_path.get_xypath_forplot()
            xy = self.selected_recorded_path.get_xypath()
            print(xy[0])
            print(xy[1])
            plt.figure()
            plt.grid()
            plt.axis("equal")
            plt.plot(x,y, "xg")
            plt.xlabel('x(m)')
            plt.ylabel('y(m)')
            plt.title(self.selected_recorded_path.get_filename())

        else:
            print('ERROR: No planned path loaded')
    
    def plot_task_and_record(self, task_name, record_name):
        print("plotting")
        task:TaskPath
        record:RecordedPath
        task = self.task_paths.get(task_name)
        record = self.recorded_paths.get(record_name)
        
        if task and record:
            plt.figure()
            plt.grid()
            plt.axis("equal")

            
            task_lats, task_lons = task.get_latlon_path_forplot()
            record_lats, record_lats = record.get_latlon_path_forplot()
            
            ref_lat, ref_lon = record_lats[50], record_lats[50]
            
            task_xy = tools.ll2xy(ref_lat, ref_lon, np.array(task_lats), np.array(task_lons))
            record_xy = tools.ll2xy(ref_lat, ref_lon, np.array(record_lats), np.array(record_lats))
            
            num = 10
            print(record_xy[0][:num], record_xy[1][:num])

            plt.plot(task_xy[0], task_xy[1], "xg")
            plt.plot(record_xy[0], record_xy[1], "-xb")
            plt.xlabel('x(m)')
            plt.ylabel('y(m)')
            
            plt.title(f"Task vs Record")
            plt.legend((f"task:{task_name}", f"record:{record_name}"))
            
        else:
            task_found = True if task else False
            record_found = True if record else False
            print(f"task found: {task_found} record found: {record_found}")
    
    def _plot_all(self, path):
        print('not implemented')
    
    def list_all_taskpaths(self):
        print("TASKS STORED IN PATH PLOTTER:")
        for k in self.task_paths.keys():
            print(">",k)


    def list_all_recordpaths(self):
        print(self.recorded_paths.keys())
    
    def delete_all_taskpaths(self):
        self.task_paths.clear()
        self.selected_task_path = None
        print("task paths cleared")
    
    def delete_all_recordedpaths(self):
        self.recorded_paths.clear()
        self.selected_record_path = None
        print("recorded paths cleared")