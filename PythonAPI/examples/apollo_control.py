#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import threading
#from multiprocessing import Process, Pipe

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

import zmq
import time

import cv2

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, sender, receiver, control, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.sender = sender
        self.receiver = receiver
        self.control = control
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = 'vehicle.lincoln.mkz2017'
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.receiver_thread = None

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice()
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        #self.camera_manager.spawn_lidar()
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        
        
        #self.parent_conn, self.child_conn = Pipe()
        #self.receiver_p = Process(target= self.receiver.tick, args=(self.child_conn,))
        #self.receiver_p.daemon = True
        #self.receiver_p.start()
        
        self.receiver_lock = threading.Lock()
        self.receiver_t = threading.Thread(target= self.receiver.tick, args=(self,))
        self.receiver_t.daemon = True
        self.receiver_t.start()
        #self.receiver_t.join()
        
        self.control_lock = threading.Lock()
        self.control_t = threading.Thread(target= self.control.applyControl, args=(self,))
        self.control_t.daemon = True
        self.control_t.start()
        #self.control_t.join()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)
        self.sender.tick(self, clock)

        #print(self.receiver._multipart_string, "\n\n\n")
        #try:
            #print(self.receiver._trajectory_list[0], "\n")
        #except:
            #print(self.receiver._trajectory_list, "\n")
        #print(len(self.receiver._multipart_list), "\n")
        #print(self.receiver._multipart_list[0], " ", self.receiver._multipart_list[1], " ", self.receiver._multipart_list[2], " ", self.receiver._multipart_list[3], "\n")
        #print(self.receiver._multipart_list[ 0 : 100], "\n")
        #print(self.receiver._multipart_list[ -10 : ], "\n")
        
        #print(self.receiver._trajectory_list, "\n")
        
        #print(self.control.trajectory_points, "\n")

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                #elif event.key == K_F2:
                    #world.hud.notification("Position lights")
                    #world.receiver.enabled = not world.receiver.enabled
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.lidar = None
        self.image_data_array = None
        self.width = None
        self.h_fov = None
        #self.encoding = 'bgra8'
        self.points = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('rotation_frequency', '20')
                bp.set_attribute('channels', '32')
                bp.set_attribute('range', '100')
                bp.set_attribute('upper_fov', '15.0')
                bp.set_attribute('lower_fov', '-25.0')
                bp.set_attribute('points_per_second', '56000')
                self.lidar_range = 10

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            if self.lidar is not None:
                self.lidar.destroy()
                #self.spawn_lidar()
  
            #if (index == 8):
                #self.spawn_lidar()
                #weak_self = weakref.ref(self)
                
                #self.lidar.listen(lambda image: CameraManager._parse_image(weak_self, image))
            #else:
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            
            #self.lidar = self._parent.get_world().spawn_actor(
            #self.sensors[-1][-1],
            #carla.Transform(carla.Location(0, 0, 2), carla.Rotation()),
            ##['sensor.lidar_parse_image.ray_cast', None, 'Lidar (Ray-Cast)']
            #attach_to=self._parent,
            #attachment_type=carla.AttachmentType.Rigid
            #)
            self.spawn_lidar()

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            
            self.lidar.listen(lambda image: CameraManager._parse_image(weak_self, image, "lidar"))
            #if self.sensors[self.index][0].startswith('sensor.camera.rgb'): 
                #self.spawn_lidar()
                #self.lidar.listen(lambda image: CameraManager._parse_image(weak_self, image))
                
            
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def spawn_lidar(self):
        self.lidar = self._parent.get_world().spawn_actor(
            self.sensors[-1][-1],
            carla.Transform(carla.Location(0, 0, 2), carla.Rotation()),
            #['sensor.lidar_parse_image.ray_cast', None, 'Lidar (Ray-Cast)']
            attach_to=self._parent,
            attachment_type=carla.AttachmentType.Rigid
        )

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, test='camera'):
        self = weak_self()
        if not self:
            return
        
        #if self.sensors[self.index][0].startswith('sensor.camera.rgb'):

            #points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            #points = np.reshape(points, (int(points.shape[0] / 4), 4))
            #points = points[..., [0, 1, 2, 3]]
            
            ##points = np.rot90(points)
            
            #for point in points:
                ##point[0] = -point[0]
                #point[1] = -point[1]
                ##point[2] = -point[2]
                
            #self.points = points
            
            
        if self.sensors[self.index][0].startswith('sensor.lidar'):

            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            points = points[..., [0, 1, 2, 3]]
            
            #points = np.rot90(points)
            
            for point in points:
                #point[0] = -point[0]
                point[1] = -point[1]
                #point[2] = -point[2]
                
            self.points = points
            
            #self.points = - points
                
            #lidar_data = np.array(points[:, :2])
            #lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            #lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            #lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            #lidar_data = lidar_data.astype(np.int32)
            #lidar_data = np.reshape(lidar_data, (-1, 2))
            #lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            #lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            #lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

            #self.surface = pygame.surfarray.make_surface(lidar_img)
            
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            
            if (test == "lidar"):
                points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                points = points[..., [0, 1, 2, 3]]
                
                #points = np.rot90(points)
                
                for point in points:
                    #point[0] = -point[0]
                    point[1] = -point[1]
                    #point[2] = -point[2]
                    
                self.points = points
                
            else:
                
                self.image_data_array = np.ndarray(
                    shape=(image.height, image.width, 4),
                    dtype=np.uint8, buffer=image.raw_data)
                
                self.width = float(image.width)
                self.h_fov = float(image.fov) * np.pi / 180.0
                    
                image.convert(self.sensors[self.index][1])
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- ØMQ sender + receiver----------------------------------------------------
# ==============================================================================
class Receiver(object):
    def __init__(self):
       
        
        #try:
            #raw_input
        #except NameError:
            #Python 3
            #raw_input = input

        self.context = zmq.Context()
        
        # Socket to receive messages on
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("tcp://*:5559")
        time.sleep(0.2)
        
        self._multipart_string = ""
        self._multipart_list = []
        self._trajectory_list = []
        self.trajectory_point_len = 10
        self.path_point_len = 7
        self._timestamp_sec = ""
    

    #def tick(self, conn):
        
        #while True:
            ##with world.receiver_lock:
            #_multipart_string = self.receiver.recv().decode("utf-8")
            ##conn.send(_multipart_string)
            ##print(self.receiver.recv().decode("utf-8"))
            #print("test")
                ##print(self._multipart_string)
        #return
    
    def tick(self, world):
        
        while True:
            with world.receiver_lock:
                
                self._multipart_string = self.receiver.recv().decode("utf-8")
                self._multipart_list = self._multipart_string.split("\n")
                self._multipart_list.pop()
                
                #for i in range(int(len(self._multipart_list) / self.trajectory_point_len)) :
                    
                    #temp_list = self._multipart_list[ self.trajectory_point_len * i : (self.trajectory_point_len * i) + self.trajectory_point_len]
                    
                    ##print(self._multipart_list[ self.trajectory_point_len * i : (self.trajectory_point_len * i) + self.trajectory_point_len])
                    
                    ##try:
                    #self._trajectory_list[i].append(temp_list)
                    #except:
                        #print(temp_list)
                    
                    #for j in range(self.trajectory_point_len - self.path_point_len) : 
                        #self._trajectory_list[i].append(self._multipart_list[(self.trajectory_point_len * i) + self.path_point_len + j])
                    
                    #self._trajectory_list[i].append(self._multipart_list[ (self.trajectory_point_len * i) + (self.trajectory_point_len - self.gaussian_info_len) : self.trajectory_point_len * (i + 1)])
                
                #conn.send(_multipart_string)
                #print(self._multipart_string, "\n")
                #print(self.receiver.recv().decode("utf-8"))
        return    

    def printString(self, world):
        
        print(self._multipart_string)


class Sender(object):
    def __init__(self):
       
        
        #try:
            #raw_input
        #except NameError:
            #Python 3
            #raw_input = input

        self.context = zmq.Context()
        
        # Socket to send messages on
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind("tcp://*:5556")
        time.sleep(0.2)
        
        # Socket with direct access to the sink: used to synchronize start of batch
        self.sink = self.context.socket(zmq.PUSH)
        self.sink.connect("tcp://localhost:5557")
        
        self.old_c = [0, 1]
        self.new_c = [0, 1]
        
        #NewValue = (((OldValue - self.old_c[0]) * (self.new_c[1] - self.new_c[0])) / (self.old_c[1] - self.old_c[0])) + self.new_c[0]
        
        #print("Press Enter when the workers are ready: ")
        #_ = raw_input()
        ##print("Sending tasks to workers...")
        
        # The first message is "0" and signals start of batch
        #sink.send(b'0')
        
        #self._multipart_message = []
        self._multipart_string = ""
        self._server_clock = pygame.time.Clock()

        self.header_time = 0
    #def on_world_tick(self, timestamp):
        #self._server_clock.tick()
        #self.server_fps = self._server_clock.get_fps()
        #self.frame = timestamp.frame
        #self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        #self._notifications.tick(world, clock)
        #if not self._show_info:
            #return
        t = world.player.get_transform()
        s = world.player.get_location()
        v = world.player.get_velocity()
        av = world.player.get_angular_velocity()
        ac = world.player.get_acceleration()
        c = world.player.get_control()
        p = world.player.get_physics_control()
        l = world.player.get_light_state()
        r = world.camera_manager.points
        i = world.camera_manager.image_data_array
        key_pressed = pygame.key.get_pressed()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        #colhist = world.collision_sensor.get_collision_history()
        #collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        #max_col = max(1.0, max(collision))
        #collision = [x / max_col for x in collision]  
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        #========================================================================
        # -- Chassis data 
        #========================================================================
        
        # engine_started
        self._multipart_string += "{}\n".format('true')
        # engine_rpm
        self._multipart_string += "{:.0f}\n".format(((3.36 * 2.237 * math.sqrt(v.x**2 + v.y**2 + v.z**2) * p.forward_gears[c.gear-1].ratio * 336.13) / (p.wheels[0].radius * 2 * 0.3937)))
        # speed_mps
        self._multipart_string += "{:.6f}\n".format((math.sqrt(v.x**2 + v.y**2 + v.z**2)))
        # odometer_m
        self._multipart_string += "{}\n".format('0')
        # fuel_range_m
        self._multipart_string += "{}\n".format('0')
        # throttle_percentage
        self._multipart_string += "{:.6f}\n".format((c.throttle * 100))
        # brake_percentage
        self._multipart_string += "{:.6f}\n".format((c.brake * 100))
        # steering_percentage
        self._multipart_string += "{:.6f}\n".format(-(c.steer * 100))
        
        if key_pressed[K_LEFT] or key_pressed[K_a]:
            
            # steering_torque_nm
            self._multipart_string += "{}\n".format('-1')
            
        elif key_pressed[K_RIGHT] or key_pressed[K_d]:
            
            # steering_torque_nm
            self._multipart_string += "{}\n".format('1')
            
        else:
            
            # steering_torque_nm
            self._multipart_string += "{}\n".format('0')
            
        # parking_brake
        self._multipart_string += "{}\n".format('false')
        # wiper
        self._multipart_string += "{}\n".format('false')
        # driving_mode
        self._multipart_string += "{}\n".format('COMPLETE_AUTO_DRIVE')
        ## error_code
        #self._multipart_string += "{}\n".format('NO_ERROR')
        
        try:
            if (c.gear > 0):
                
                #gear_location
                self._multipart_string += "{}\n".format('GEAR_DRIVE')
                
            elif (c.gear == 0):
                
                #gear_location
                self._multipart_string += "{}\n".format('GEAR_NEUTRAL')
                
            else:
                
                #gear_location
                self._multipart_string += "{}\n".format('GEAR_REVERSE')

        except:
            if (c.gear > '0'):
                
                #gear_location
                self._multipart_string += "{}\n".format('GEAR_DRIVE')
                
            elif (c.gear == '0'):
                
                #gear_location
                self._multipart_string += "{}\n".format('GEAR_NEUTRAL')
                
            else:
                
                #gear_location
                self._multipart_string += "{}\n".format('GEAR_REVERSE') 
        
        # steering_timestamp
        self._multipart_string += "{:.6f}\n".format(time.time())
        
        # chassis_header
        
            # timestamp_sec
        self._multipart_string += "{:.6f}\n".format(time.time())
            # timestamp_sec
        self._multipart_string += "{}\n".format('"chassis"')
        
        # l.RightBlinker
        if (False):
            
            # turn_signal
            self._multipart_string += "{}".format('TURN_RIGHT')
            
        # l.LeftBlinker    
        elif (False):
            
            # turn_signal
            self._multipart_string += "{}\n".format('TURN_LEFT')
            
        else:
            
            # turn_signal
            self._multipart_string += "{}\n".format('TURN_NONE') 
        
        # l.HighBeam
        if (False):
            
            # high_beam
            self._multipart_string += "{}\n".format('true') 
        
        else:
            
            # high_beam
            self._multipart_string += "{}\n".format('false')
            
        # l.LowBeam    
        if (False):
            
            # low_beam 
            self._multipart_string += "{}\n".format('true')
            
        else:
            
            # low_beam
            self._multipart_string += "{}\n".format('false')
            
        # horn
        self._multipart_string += "{}\n".format('false')  
        
        # l.Position    
        if (False):
            
            # emergency_light  
            self._multipart_string += "{}\n".format('true')
            
        else:
            
            # emergency_light 
            self._multipart_string += "{}\n".format('false')
        
        
        #========================================================================
        # -- LocalizationEstimate data
        #========================================================================
        
        # chassis_header
        
            # timestamp_sec
        self._multipart_string += "{:.6f}\n".format(time.time())
            # frame_id
        self._multipart_string += "{}\n".format('"novatel"')
        
        # position
        
        # x
        self._multipart_string += "{:.14f}\n".format(t.location.x)
        # y
        self._multipart_string += "{:.15f}\n".format(-t.location.y)
        # z
        self._multipart_string += "{:.15f}\n".format(t.location.z)
       
        # orientation
        
        # qx qy qz qw
        self._multipart_string += "{:.20f}\n{:.20f}\n{:.20f}\n{:.20f}\n".format(*euler_to_quaternion(t.rotation.roll, t.rotation.pitch, t.rotation.yaw))
        
        # linear_velocity
        
        # x
        self._multipart_string += "{:.15f}\n".format(v.x)
        # y
        self._multipart_string += "{:.15f}\n".format(v.y)
        # z
        self._multipart_string += "{:.15f}\n".format(v.z)
       
        ## linear_acceleration
        
        ## x y z
        #self._multipart_string += "{:.18f}\n{:.18f}\n{:.18f}\n".format(*world.imu_sensor.accelerometer)
        
        ## angular_velocity
        
        ## x y z
        #self._multipart_string += "{:.18f}\n{:.18f}\n{:.18f}\n".format(*world.imu_sensor.gyroscope)
        
        # heading
        
        self._multipart_string += "{:.15f}\n".format(- math.radians(compass) + math.pi/2)
        
        # linear_acceleration_vrf
        
        # x
        self._multipart_string += "{:.15f}\n".format(ac.x)
        # y
        self._multipart_string += "{:.15f}\n".format(ac.y)
        # z
        self._multipart_string += "{:.15f}\n".format(ac.z)
        
        # angular_velocity_vrf
        
        # x
        self._multipart_string += "{:.15f}\n".format(av.x)
        # y
        self._multipart_string += "{:.15f}\n".format(av.y)
        # z
        self._multipart_string += "{:.15f}\n".format(av.z)
        
        # euler angles
        
        # x
        self._multipart_string += "{:.15f}\n".format(math.radians(t.rotation.roll))
        # y
        self._multipart_string += "{:.15f}\n".format(math.radians(t.rotation.pitch))
        # z
        self._multipart_string += "{:.15f}\n".format(math.radians(t.rotation.yaw))
        
        # measurement_time
        self._multipart_string += "{:.6f}\n".format(time.time())
        
        ##========================================================================
        ## -- TransformStamped data
        ##========================================================================
        
        ## tf_header
            ## timestamp_sec
        #self._multipart_string += "{:.6f}\n".format(time.time())
            ## frame_id
        #self._multipart_string += "{}\n".format('"world"')  
        
        ## repeated lenght
        #self._multipart_string += "{}\n".format(1)
        
        ## tf_header
            ## timestamp_sec
        #self._multipart_string += "{:.6f}\n".format(time.time())
            ## frame_id
        #self._multipart_string += "{}\n".format('"world"')  
        
        ## child_frame_id
        #self._multipart_string += "{}\n".format('"localization"')
        ## translation
            ## x
        #self._multipart_string += "{:.14f}\n".format(t.location.x)
            ## y
        #self._multipart_string += "{:.15f}\n".format(-t.location.y)
            ## z
        #self._multipart_string += "{:.15f}\n".format(t.location.z)
        
        ## rotation
            ## qx qy qz qw
        #self._multipart_string += "{:.20f}\n{:.20f}\n{:.20f}\n{:.20f}\n".format(*euler_to_quaternion(t.rotation.roll, t.rotation.pitch, t.rotation.yaw))
        
        ##========================================================================
        ## -- Lidar data
        ##========================================================================
        
        #if r is not None:
            
            ##measurement_time = time.time()
            ##for lidar_point in lidar_data:
                ###cyber_point = PointXYZIT()
                ##print("[{}\n{}\n{}\n p:{}]\n".format(-lidar_point[0], -lidar_point[1], -lidar_point[2], lidar_point[3]))
                ###cyber_point.x = lidar_point[0]
                ###cyber_point.y = lidar_point[1]
                ###cyber_point.z = lidar_point[2]

            ## chassis_header
                ## timestamp_sec
            #self._multipart_string += "{:.6f}\n".format(time.time())
                ## frame_id
            #self._multipart_string += "{}\n".format('"lidar128"')    
            ## frame_id
            #self._multipart_string += "{}\n".format('"lidar128"')
            ## repeated lenght
            #self._multipart_string += "{}\n".format(len(r))
            
            ##print(len(r))
                
            #for point in r:    
                
                ##print("{:.0f}".format((((point[3] - self.old_c[0]) * (self.new_c[1] - self.new_c[0])) / (self.old_c[1] - self.old_c[0])) + self.new_c[0]))
                
                ##self._multipart_string += "{:.5f}\n{:.5f}\n{:.5f}\n{:.0f}\n".format(point[0], point[1], point[2], (((point[3] - self.old_c[0]) * (self.new_c[1] - self.new_c[0])) / (self.old_c[1] - self.old_c[0])) + self.new_c[0])
                
                #self._multipart_string += "{:.5f}\n{:.5f}\n{:.5f}\n".format(point[0], point[1], point[2])
                
           
            ## measurement_time
            #self._multipart_string += "{:.6f}\n".format(time.time())
                ###self.msg.point.append(cyber_point)
            ##print(len(r))
            ##dt = measurement_time - self.send_time
            ##if dt >= seconds_per_rotation:
                ##print(len(lidar_data))
             
            ## width, height
            #self._multipart_string += "{}\n{}\n".format(1, len(r))
    
        ##print(world.camera_manager.sensors[6].)    
        
        
        ##self._multipart_string = self._multipart_string[:-1]
        
        ##========================================================================
        ## -- Camera data
        ##========================================================================
        
        #if i is not None:
            
            ##x = len(i[0]) / 2.0
            ##y = len(i) / 2.0
            ##f = world.camera_manager.width / 2.0 / np.tan(world.camera_manager.h_fov / 2.0)
            ##measurement_time = time.time()
            ##for lidar_point in lidar_data:
                ###cyber_point = PointXYZIT()
                ##print("[{}\n{}\n{}\n p:{}]\n".format(-lidar_point[0], -lidar_point[1], -lidar_point[2], lidar_point[3]))
                ###cyber_point.x = lidar_point[0]
                ###cyber_point.y = lidar_point[1]
                ###cyber_point.z = lidar_point[2]
                
            ##COMPRESSED_IMAGE
                
            ## chassis_header
                ## timestamp_sec
            #self._multipart_string += "{:.6f}\n".format(time.time())
                ## frame_id
            #self._multipart_string += "{}\n".format('"front_6mm"')    
            ## frame_id
            #self._multipart_string += "{}\n".format('"front_6mm"')
            ## format
            #self._multipart_string += "{}\n".format('"jpeg"')
            ## data
            #self._multipart_string += "{}\n".format(str(cv2.imencode('.jpg', i)[1].tostring())[1:])
            ##print(str(cv2.imencode('.jpg', i)[1].tostring())[1:])
            ##repeated lenght
            
            ## measurement_time
            #self._multipart_string += "{:.6f}\n".format(time.time())
            
            ##self._multipart_string += "{}\n".format(len(r))
            
            ##IMAGE_POINT_CLOUD
            
            ### chassis_header
                ### timestamp_sec
            ##self._multipart_string += "{:.6f}\n".format(time.time())
                ### frame_id
            ##self._multipart_string += "{}\n".format('"front_6mm"')    
            ### frame_id
            ##self._multipart_string += "{}\n".format('"front_6mm"')
                
            ##for u in range(len(i[0])):
                ##for v in range(len(i)):    
                
                    ##point_x = i[v][u]
                    ##print(len(i[0]))
                    ##print(len(i))
                    
                    ##self._multipart_string += "{:.5f}\n{:.5f}\n{:.5f}\n".format(point_x, float(-(u - x) * point_x / f), float(-(v - y) * point_x / f))
                    
        ##all_chars = set(chr(i) for i in range(256))
        ##unused_chars = all_chars - set(self._multipart_string)
        ##print(unused_chars,"\n")  
        
        #========================================================================
        # -- Perception_obstacles data
        #========================================================================
        
        # repeated lenght
        self._multipart_string += "{}\n".format(len(world.world.get_actors().filter('vehicle.*')) + len(world.world.get_actors().filter('walker.pedestrian.*')) - 1 )
        
        for actor in world.world.get_actors().filter('vehicle.*'):
            
            if actor.id != world.player.id:
                #print(actor, ", ")
                #print(actor.bounding_box.location, ", ")
                #print(actor.bounding_box.extent, ", ")
                #print(actor.id, ", ")
                #print(world.player.id, ", ")
                #print(actor.get_transform().location, ", ")
                #print("\n")
                
                # PERCEPTION OBSTACLE (VEHICLE)
                # id
                self._multipart_string += "{}\n".format(actor.id) 
                # position
                    # x
                self._multipart_string += "{}\n".format(actor.get_transform().location.x) 
                    # y
                self._multipart_string += "{}\n".format(-actor.get_transform().location.y) 
                    # z
                self._multipart_string += "{}\n".format(actor.get_transform().location.z) 
                # theta
                self._multipart_string += "{}\n".format(-math.radians(actor.get_transform().rotation.yaw))
                # velocity
                    # x
                self._multipart_string += "{}\n".format(actor.get_velocity().x) 
                    # y
                self._multipart_string += "{}\n".format(-actor.get_velocity().y) 
                    # z
                self._multipart_string += "{}\n".format(actor.get_velocity().z) 
                # length
                self._multipart_string += "{}\n".format(actor.bounding_box.extent.x * 2.0) 
                # width
                self._multipart_string += "{}\n".format(actor.bounding_box.extent.y * 2.0)  
                # height
                self._multipart_string += "{}\n".format(actor.bounding_box.extent.z * 2.0)
                # type
                self._multipart_string += "{}\n".format("VEHICLE")
        
        for actor in world.world.get_actors().filter('walker.pedestrian.*'):
            #print(actor, ", ")
            #print(actor.bounding_box.location, ", ")
            #print(actor.bounding_box.extent, ", ")
            #print(actor.id, ", ")
            #print(world.player.id, ", ")
            #print(actor.get_transform().location, ", ")
            #print("\n")
            
            # PERCEPTION OBSTACLE (VEHICLE)
            # id
            self._multipart_string += "{}\n".format(actor.id) 
            # position
                # x
            self._multipart_string += "{}\n".format(actor.get_transform().location.x) 
                # y
            self._multipart_string += "{}\n".format(-actor.get_transform().location.y) 
                # z
            self._multipart_string += "{}\n".format(actor.get_transform().location.z) 
            # theta
            self._multipart_string += "{}\n".format(-math.radians(actor.get_transform().rotation.yaw))
            # velocity
                # x
            self._multipart_string += "{}\n".format(actor.get_velocity().x) 
                # y
            self._multipart_string += "{}\n".format(-actor.get_velocity().y) 
                # z
            self._multipart_string += "{}\n".format(actor.get_velocity().z) 
            # length
            self._multipart_string += "{}\n".format(actor.bounding_box.extent.x * 2.0) 
            # width
            self._multipart_string += "{}\n".format(actor.bounding_box.extent.y * 2.0)  
            # height
            self._multipart_string += "{}\n".format(actor.bounding_box.extent.z * 2.0)
            # type
            self._multipart_string += "{}\n".format("PEDESTRIAN")
            
            #print(actor.bounding_box.extent.x * 2.0, actor.bounding_box.extent.y * 2.0)
            
        # perception_obstacle_header
            # timestamp_sec
        self._multipart_string += "{:.6f}\n".format(time.time())
            # frame_id
        self._multipart_string += "{}\n".format('"third_party_perception"')
        
        
        #========================================================================
        # -- Send data
        #========================================================================
                    
        self.header_time = time.time()
            
        self.socket.send_string(self._multipart_string)
        
        self._multipart_string = ""

class ApolloControl(object):
    def __init__(self):
       
        self.height_buffer = 0.1
        self.trajectory_list = []
        self.trajectory_points = []
        
    def getApproximatePoint(self, points_list, reference_point):
        distance_list = []
        # you could remove the sqrt for computation benefits, its a symetric func
        # that does not change the relative ordering of distances
        for point in points_list:
            distance_list.append(math.sqrt((reference_point[0] - point[0]) ** 2 + (reference_point[1] - point[1]) ** 2)) 
            
        return distance_list.index(min(distance_list))
    
    def applyControl(self, world):
        
        world.player.set_simulate_physics(False)
        
        while True:
            with world.control_lock:
                timestamp = time.time()   
                self.transform = world.player.get_transform()
                wp = world.map.get_waypoint(self.transform.location)
                wait_time = 0
                start_time = 0
                n = 0
                reference_point = 0
                old_tp = 0
                
                #self.trajectory_points = world.receiver._trajectory_list
                #for tp in self.trajectory_points:
                    #if dt < tp[3]:

                        
                        #self.transform.location.x = tp[0][0]
                        #self.transform.location.y = -tp[0][1]
                        #self.transform.location.z = wp.transform.location.z + self.height_buffer
                        #self.transform.rotation.yaw = -math.degrees(tp[0][3])
                        #world.player.set_transform
                self.trajectory_list = world.receiver._multipart_list
                
                #timestamp_sec = self.trajectory_list[0]
                
                #print(timestamp_sec)
                if self.trajectory_list:
                    
                    self.header = self.trajectory_list[0:16]
                    
                    dt = timestamp - float(self.header[0])
                    start_point = dt
                    
                    #del self.trajectory_list[0:16]
                    
                    self.trajectory_points = [self.trajectory_list[x + 16:x + 26] for x in range(0, len(self.trajectory_list), 10)]
                    
                    #print(self.trajectory_points[-10:0])
                    
                    del self.trajectory_points[-2:]
                    
                    #print(self.trajectory_points[-10:0])
                    
                    #try:
                    self.location_points = [[float(self.trajectory_points[x][0]), -float(self.trajectory_points[x][1])] for x in range(0, len(self.trajectory_points))]
                    #except:
                        #print("ERRO", len(self.trajectory_list))
                    
                    reference_point = [self.transform.location.x, self.transform.location.y]
                    n = self.getApproximatePoint(self.location_points, reference_point)
                    
                    #print(reference_point, n, self.location_points[n-1], self.location_points[n], self.location_points[n+1])
                    
                    #print(self.trajectory_points[:20])
                    if self.trajectory_points:
                        #print("PASSED")  
                        count = 0
                        for tp in self.trajectory_points:
                            
                            #for i in range(n):
                                #continue
                            if count < (n + 1):
                                #print(count)
                                if count == n:
                                    old_tp = tp
                                count += 1  
                                continue
                            
                            else:
                                #print("PASSED")
                                
                                if dt < float(tp[9]):
                                    
                                    #del self.trajectory_points[0:self.trajectory_points.index(tp)]
                                    
                                    #for tp in self.trajectory_points:
                                    
                                    start_time = time.time()
                                    
                                    if start_point == dt:
                                        start_point = float(old_tp[9])
                                        
                                    while wait_time < (float(tp[9]) - start_point):
                                        wait_time = time.time() - start_time
                                      
                                    print(dt, wait_time, float(tp[9]))
                                    
                                    self.transform.location.x = float(tp[0])
                                    self.transform.location.y = -float(tp[1])
                                    self.transform.location.z = wp.transform.location.z + self.height_buffer
                                    self.transform.rotation.yaw = -math.degrees(float(tp[2]))
                                    world.player.set_transform(self.transform)
                                    
                                    #if self.trajectory_list is not world.receiver._multipart_list:
                                        #break
                                    
                                    start_point += wait_time
                                    wait_time = 0
                                    
                                    
                                    
                                    #dt_start = dt
                                    #while self.trajectory_list is world.receiver._multipart_list:
                                        #loop_now = time.time()
                                        #wait_time = dt + (loop_now - loop_starts)
                                        #if wait_time >= float(tp[9]):
                                            #print(dt, wait_time, float(tp[9]))
                                            #break
                                    
                                    
                                    
 
                            #if self.trajectory_list is not world.receiver._multipart_list:
                                #break
                            
                        #if self.trajectory_list is not world.receiver._multipart_list:
                            #print("CHANGED")
                            #break         
        return    

    def printString(self, world):
        
        print(self._multipart_string)        
    
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        sender = Sender()
        receiver = Receiver()
        control = ApolloControl()
        world = World(client.get_world(), hud, sender, receiver, control, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
