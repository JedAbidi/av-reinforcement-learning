# carla_env_state.py - ROBUST VERSION with better spawn handling

import gym
from gym import spaces
import numpy as np
import carla
import random
import time

class CarlaEnvState(gym.Env):
    def __init__(self, host='localhost', port=2000):
        super(CarlaEnvState, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self.vehicle = None
        self.collision_sensor = None
        self.collision_hist = []
        
        # Observation: [speed, steer, throttle, brake, lane_offset, heading_error, next_wp_x, next_wp_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # Action: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.episode_step = 0
        self.last_location = None
        print("State-based CARLA env ready (ROBUST VERSION)")
    
    def reset(self):
        """Reset with robust spawn handling"""
        # Cleanup existing actors
        self._cleanup()
        
        # Wait for CARLA to process cleanup
        time.sleep(0.1)
        self.world.tick()
        time.sleep(0.1)
        
        # Try spawning with retries
        self.vehicle = self._spawn_vehicle_safe()
        self.collision_sensor = self._spawn_collision_sensor()
        
        self.collision_hist = []
        self.episode_step = 0
        self.last_location = self.vehicle.get_location()
        
        # Warm up
        for _ in range(5):
            self.world.tick()
        
        return self._get_obs()
    
    def _spawn_vehicle_safe(self, max_retries=10):
        """Spawn vehicle with retry logic to avoid collision errors"""
        bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        for attempt in range(max_retries):
            try:
                # Pick random spawn point
                spawn = random.choice(spawn_points)
                
                # Try to spawn
                vehicle = self.world.spawn_actor(bp, spawn)
                
                # Success!
                return vehicle
                
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    # Spawn point blocked, try another
                    if attempt < max_retries - 1:
                        # Wait a bit and try again
                        time.sleep(0.2)
                        self.world.tick()
                        continue
                    else:
                        # Last attempt failed, clear all vehicles and retry once more
                        print(f"⚠️ Spawn failed {max_retries} times, clearing world...")
                        self._clear_all_vehicles()
                        time.sleep(0.5)
                        for _ in range(3):
                            self.world.tick()
                        
                        # Final attempt with first spawn point
                        spawn = spawn_points[0]
                        vehicle = self.world.spawn_actor(bp, spawn)
                        return vehicle
                else:
                    raise e
        
        # Should never reach here
        raise RuntimeError("Failed to spawn vehicle after all retries")
    
    def _clear_all_vehicles(self):
        """Emergency cleanup of all vehicles in world"""
        try:
            vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                try:
                    vehicle.destroy()
                except:
                    pass
        except:
            pass
    
    def step(self, action):
        self.episode_step += 1
        
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
        self.world.tick()
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = len(self.collision_hist) > 0 or self.episode_step >= 2000
        info = self._get_info()
        
        return obs, reward, done, info
    
    def _spawn_collision_sensor(self):
        """Spawn collision sensor"""
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        sensor.listen(lambda e: self.collision_hist.append(e))
        return sensor
    
    def _get_obs(self):
        vel = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        ctrl = self.vehicle.get_control()
        
        loc = self.vehicle.get_location()
        wp = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_offset = loc.distance(wp.transform.location) if wp else 5.0
        
        veh_yaw = self.vehicle.get_transform().rotation.yaw
        wp_yaw = wp.transform.rotation.yaw if wp else veh_yaw
        heading_error = (wp_yaw - veh_yaw + 180) % 360 - 180
        
        nxt = wp.next(10.0)[0] if wp else wp
        next_x = (nxt.transform.location.x - loc.x) if nxt else 0.0
        next_y = (nxt.transform.location.y - loc.y) if nxt else 0.0
        
        return np.array([speed, ctrl.steer, ctrl.throttle, ctrl.brake, lane_offset, heading_error/180, next_x/10, next_y/10], dtype=np.float32)
    
    def _compute_reward(self):
        """Fixed reward: penalizes staying still, rewards forward motion"""
        if len(self.collision_hist) > 0:
            return -100.0
        
        vel = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        loc = self.vehicle.get_location()
        wp = self.world.get_map().get_waypoint(loc, project_to_road=True)
        lane_offset = loc.distance(wp.transform.location) if wp else 10.0
        
        # Distance traveled this step
        current_location = self.vehicle.get_location()
        distance_reward = 0.0
        if self.last_location is not None:
            distance_traveled = current_location.distance(self.last_location)
            distance_reward = distance_traveled * 10.0
        self.last_location = current_location
        
        # Speed component
        if speed < 5.0:
            speed_reward = -10.0  # Harsh penalty for being stationary
        elif speed < 15.0:
            speed_reward = speed * 0.5
        elif speed < 30.0:
            speed_reward = speed * 1.0
        else:
            speed_reward = 30.0 - (speed - 30.0) * 0.5
        
        # Lane keeping
        lane_penalty = -lane_offset * 3.0
        
        return distance_reward + speed_reward + lane_penalty
    
    def _get_info(self):
        vel = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        loc = self.vehicle.get_location()
        wp = self.world.get_map().get_waypoint(loc, project_to_road=True)
        lane_offset = loc.distance(wp.transform.location) if wp else 10.0
        
        return {
            'speed_kmh': speed,
            'collisions': len(self.collision_hist),
            'episode_step': self.episode_step,
            'lane_offset': lane_offset
        }
    
    def _cleanup(self):
        """Cleanup actors safely"""
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None
        
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass
            self.vehicle = None
    
    def close(self):
        self._cleanup()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except:
            pass
