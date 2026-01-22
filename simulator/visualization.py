"""
Visualization Server

WebSocket server that streams aircraft state to a Three.js frontend.
Also provides HTTP serving for the frontend files.
"""

import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any, Callable
from dataclasses import asdict
import http.server
import socketserver
import threading
from pathlib import Path

try:
    import websockets
    from websockets.server import serve as ws_serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Warning: websockets not installed. Run: pip install websockets")

from .state import AircraftState, ControlInputs
from .dynamics import FlightDynamics


class StateEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)


def state_to_message(state: AircraftState, forces_moments=None, crash_state=None) -> str:
    """
    Convert aircraft state to JSON message for frontend.
    
    Returns position, quaternion, and derived values for Three.js.
    """
    phi, theta, psi = state.euler_angles
    
    # Get rotation matrix from NED body frame quaternion
    R_body_to_ned = state.quaternion.to_dcm()
    
    # Transform rotation matrix to Three.js world coordinates
    # NED: X=North, Y=East, Z=Down
    # Three.js: X=East, Y=Up, Z=North
    # Transformation matrix: [E, -D, N] = [0 1 0; 0 0 -1; 1 0 0] * [N, E, D]
    P = np.array([
        [0, 1, 0],   # Three.js X = NED Y (East)
        [0, 0, -1],  # Three.js Y = -NED Z (Up)
        [1, 0, 0]    # Three.js Z = NED X (North)
    ])
    
    # Transform the rotation matrix: R_threejs = P * R_ned * P^T
    R_body_to_threejs = P @ R_body_to_ned @ P.T
    
    # Convert rotation matrix back to quaternion
    # Using Shepperd's method for numerical stability
    trace = np.trace(R_body_to_threejs)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R_body_to_threejs[2, 1] - R_body_to_threejs[1, 2]) * s
        qy = (R_body_to_threejs[0, 2] - R_body_to_threejs[2, 0]) * s
        qz = (R_body_to_threejs[1, 0] - R_body_to_threejs[0, 1]) * s
    elif R_body_to_threejs[0, 0] > R_body_to_threejs[1, 1] and R_body_to_threejs[0, 0] > R_body_to_threejs[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_body_to_threejs[0, 0] - R_body_to_threejs[1, 1] - R_body_to_threejs[2, 2])
        qw = (R_body_to_threejs[2, 1] - R_body_to_threejs[1, 2]) / s
        qx = 0.25 * s
        qy = (R_body_to_threejs[0, 1] + R_body_to_threejs[1, 0]) / s
        qz = (R_body_to_threejs[0, 2] + R_body_to_threejs[2, 0]) / s
    elif R_body_to_threejs[1, 1] > R_body_to_threejs[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_body_to_threejs[1, 1] - R_body_to_threejs[0, 0] - R_body_to_threejs[2, 2])
        qw = (R_body_to_threejs[0, 2] - R_body_to_threejs[2, 0]) / s
        qx = (R_body_to_threejs[0, 1] + R_body_to_threejs[1, 0]) / s
        qy = 0.25 * s
        qz = (R_body_to_threejs[1, 2] + R_body_to_threejs[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R_body_to_threejs[2, 2] - R_body_to_threejs[0, 0] - R_body_to_threejs[1, 1])
        qw = (R_body_to_threejs[1, 0] - R_body_to_threejs[0, 1]) / s
        qx = (R_body_to_threejs[0, 2] + R_body_to_threejs[2, 0]) / s
        qy = (R_body_to_threejs[1, 2] + R_body_to_threejs[2, 1]) / s
        qz = 0.25 * s
    
    data = {
        'type': 'state',
        'time': state.time,
        
        # Position (NED -> Three.js: X=East, Y=Up, Z=North)
        'position': {
            'x': state.p_east,
            'y': -state.p_down,  # Up is positive in Three.js
            'z': state.p_north
        },
        
        # Quaternion in Three.js frame
        'quaternion': {
            'w': qw,
            'x': qx,
            'y': qy,
            'z': qz
        },
        
        # Euler angles for HUD (degrees)
        'euler': {
            'roll': np.degrees(phi),
            'pitch': np.degrees(theta),
            'heading': np.degrees(psi) % 360
        },
        
        # Velocities
        'velocity': {
            'body': state.velocity_body.tolist(),
            'ned': state.velocity_ned.tolist(),
            'groundspeed': state.groundspeed,
            'climb_rate': state.climb_rate
        },
        
        # Angular rates (deg/s for display)
        'rates': {
            'p': np.degrees(state.p),
            'q': np.degrees(state.q),
            'r': np.degrees(state.r)
        },
        
        'altitude': state.altitude
    }
    
    # Add forces/moments if available
    if forces_moments is not None:
        data['aero'] = {
            'airspeed': forces_moments.airspeed,
            'alpha': np.degrees(forces_moments.alpha),
            'beta': np.degrees(forces_moments.beta),
            'dynamic_pressure': forces_moments.dynamic_pressure
        }
        data['forces'] = {
            'total': forces_moments.force.tolist(),
            'aero': forces_moments.aero_force.tolist(),
            'thrust': forces_moments.thrust_force.tolist(),
            'gravity': forces_moments.gravity_force.tolist()
        }
        data['moments'] = forces_moments.moment.tolist()
    
    # Add crash state if available
    if crash_state is not None:
        data['crash'] = {
            'crashed': crash_state.crashed,
            'crash_type': crash_state.crash_type.name if crash_state.crashed else None,
            'crash_message': crash_state.crash_message if crash_state.crashed else None,
            'stall_warning': crash_state.stall_warning,
            'overspeed_warning': crash_state.overspeed_warning,
            'terrain_warning': crash_state.terrain_warning,
            'g_force': crash_state.impact_g_force,
            'time_in_stall': crash_state.time_in_stall
        }
    
    return json.dumps(data, cls=StateEncoder)


class SimulationServer:
    """
    WebSocket server for streaming simulation state.
    
    Handles bidirectional communication:
    - Server -> Client: State updates
    - Client -> Server: Control inputs, commands
    """
    
    def __init__(
        self,
        dynamics: FlightDynamics,
        host: str = "localhost",
        port: int = 8765,
        update_rate: float = 60.0  # Hz
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets library required. Run: pip install websockets")
        
        self.dynamics = dynamics
        self.host = host
        self.port = port
        self.update_interval = 1.0 / update_rate
        
        self.clients: set = set()
        self.running = False
        self.paused = False
        
        # Control input callback
        self.control_callback: Optional[Callable[[Dict], ControlInputs]] = None
        
        # External control inputs (from WebSocket clients)
        self.external_controls: Optional[ControlInputs] = None
    
    async def register(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial state
        msg = state_to_message(
            self.dynamics.state, 
            self.dynamics.forces_moments,
            self.dynamics.crash_state
        )
        await websocket.send(msg)
    
    async def unregister(self, websocket):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_message(self, websocket, message: str):
        """Process incoming message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'control':
                # Update control inputs
                self.external_controls = ControlInputs(
                    elevator=data.get('elevator', 0.0),
                    aileron=data.get('aileron', 0.0),
                    rudder=data.get('rudder', 0.0),
                    throttle=data.get('throttle', 0.5)
                )
            
            elif msg_type == 'command':
                cmd = data.get('command', '')
                if cmd == 'reset':
                    self.dynamics.reset()
                elif cmd == 'pause':
                    self.paused = True
                elif cmd == 'resume':
                    self.paused = False
                elif cmd == 'stop':
                    self.running = False
            
            elif msg_type == 'set_state':
                # Allow client to set specific state values
                if 'altitude' in data:
                    self.dynamics.state.position[2] = -data['altitude']
                if 'airspeed' in data:
                    self.dynamics.state.velocity_body[0] = data['airspeed']
        
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")
        except Exception as e:
            print(f"Error handling message: {e}")
    
    async def client_handler(self, websocket):
        """Handle a single client connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            await self.unregister(websocket)
    
    async def simulation_loop(self):
        """Main simulation loop that steps physics and broadcasts state."""
        self.running = True
        
        while self.running:
            if not self.paused:
                # Get controls (external or default)
                controls = self.external_controls or self.dynamics.controls
                
                # Step simulation
                self.dynamics.step(controls)
                
                # Broadcast state
                msg = state_to_message(
                    self.dynamics.state, 
                    self.dynamics.forces_moments,
                    self.dynamics.crash_state
                )
                await self.broadcast(msg)
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)
    
    async def run_async(self):
        """Run the WebSocket server asynchronously."""
        async with ws_serve(self.client_handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await self.simulation_loop()
    
    def run(self):
        """Run the WebSocket server (blocking)."""
        asyncio.run(self.run_async())


def serve_frontend(directory: str, port: int = 8080):
    """
    Serve static frontend files.
    
    Args:
        directory: Path to frontend files
        port: HTTP port
    """
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        
        def log_message(self, format, *args):
            pass  # Suppress logging
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving frontend at http://localhost:{port}")
        httpd.serve_forever()


def run_with_visualization(
    dynamics: FlightDynamics,
    frontend_dir: Optional[str] = None,
    ws_port: int = 8765,
    http_port: int = 8080
):
    """
    Run simulation with visualization server.
    
    Args:
        dynamics: FlightDynamics instance
        frontend_dir: Path to frontend files (optional)
        ws_port: WebSocket port
        http_port: HTTP port for frontend
    """
    # Start HTTP server for frontend if directory provided
    if frontend_dir:
        http_thread = threading.Thread(
            target=serve_frontend, 
            args=(frontend_dir, http_port),
            daemon=True
        )
        http_thread.start()
    
    # Run WebSocket server
    server = SimulationServer(dynamics, port=ws_port)
    server.run()

