#!/usr/bin/env python3
import json
import numpy as np

class PATHTYPES():
        LOWRES_PATH = 1
        HIGHRES_PATH = 2
        path_types_dict = {
            LOWRES_PATH: "LOWRES_PATH",
            HIGHRES_PATH: "HIGHRES_PATH"
        }

def new_parse_jsontask(task_name):
    """
    Returns an array of [lat,lon] taken out of a json file with the given 'task_name'
    The file must be referenced relative to base_directory/ 
    """
    task_waypoints_array = []
    base_directory = '/app/tasks/'
    try:
        with open(base_directory + task_name) as f:
            file = json.load(f)
            
    except Exception as e:
        print(f'Problem parsing task: {e}')
        
    else:
        payload = file["payload"]
        data = payload["data"]
        vehicle_task_plan = data["vehicleTaskPlan"]
        route = vehicle_task_plan["route"]
        waypoints = route["waypoints"]
        previous_side = ""

        for i, wp in enumerate(waypoints):
            geometries = wp["geometries"]
            for j, geometry in enumerate(geometries):
                coordinates = geometry["coordinates"]
                for k, coordinate in enumerate(coordinates):
                    lat = coordinate["lat"]
                    lon = coordinate["lng"]
                    point = [lat,lon]
                    task_waypoints_array.append(point)

        # print('parsing of task done')
        return task_waypoints_array

def ll2xy(ref_lat, ref_lon, lat, lon):
        const_Re = 6371000 
        x = (lat - ref_lat) * (np.pi / 180) * const_Re
        y = (lon - ref_lon) * (np.pi / 180) * const_Re * np.cos(lat * np.pi / 180)
        
        return [x,y]