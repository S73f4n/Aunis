# Copyright (c) 2022-2025 Taner Esat <t.esat@fz-juelich.de>

import socket
import time
import numpy as np
import nanonis_spm
from scipy.optimize import curve_fit

# -------------------------------
# TCP INTERFACES
# -------------------------------
"""
Define all TCP remote interfaces here.
Key: interface name (used to reference remote devices)
Value: dict with 'host' and 'port'
    host → IP address of the remote device.
    port → TCP port where the device is listening.
"""
TCP_INTERFACES = {
    "QuPe": {"host": "192.168.1.10", "port": 1337}, # Example
}

# -------------------------------
# TCP Client Class
# -------------------------------
class TCPClient:
    """
    Encapsulates sending commands to a remote TCP interface.

    Provides two modes:
    - send: fire-and-forget
    - query: send and wait for response
    """
    def __init__(self, interface_name: str):
        if interface_name not in TCP_INTERFACES:
            raise ValueError(f"Unknown interface: {interface_name}")
        self.host = TCP_INTERFACES[interface_name]["host"]
        self.port = TCP_INTERFACES[interface_name]["port"]

    def query(self, command_name: str, args: list):
        """Sends a command with arguments to the remote device and waits for response.
        Response format expected from device: "error|response|variable"
        This format is necessary for compatibility with the Python Package for Nanonis (nanonis_spm).

        Args:
            command_name (str): Remote command name.
            args (list): Arguments for the remote command.

        Returns:
            tuple: (errorString, response, variables)
                - errorString (str): Error message.
                - response (str or bytes): Raw response from the remote device.
                - variables (list): Returned variables from the remote device.
        """
        script_line = f"{command_name} " + " ".join(map(str, args)) + "\n"
        print(script_line)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(script_line.encode())
                response_data = b""
                while True:
                    data = s.recv(1024)
                    if not data:
                        break
                    response_data += data

            # Assume the remote device sends a response in the format:
            # "error|response|variable1,variable2" (use 'None' for empty fields)
            decoded = response_data.decode().strip()
            parts = decoded.split("|")
            if len(parts) != 3:
                # Malformed response from device
                return ("Malformed response", decoded, [])
            error = "" if parts[0] == "None" else parts[0]
            response = "" if parts[1] == "None" else parts[1]
            # variable = [] if parts[2] == "None" else parts[2]
            if parts[2] == "None":
                variable = []
            else:
                variable = parts[2].split(",")
            return ([error], response, variable)

        except Exception as e:
            print(str(e))
            return (str(e), "", [])
        
    def send(self, command_name: str, args: list):
        """
        Sends a command without expecting a response.
        Returns (error, "", [])
        """
        script_line = f"{command_name} " + " ".join(map(str, args)) + "\n"
        print(script_line)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(script_line.encode())
            return ("", "", [])
        except Exception as e:
            return (str(e), "", [])
        

# -------------------------------
# Scripting Interface Class
# -------------------------------
class ScriptingInterface():
    commandList = {}

    def __init__(self):
        self.commandList = FUNCTION_REGISTRY
        self.connected = False
    
    def connect(self, ip, port): 
        """Connect to Nanonis TCP Interface.

        Args:
            ip (str): IP adress of the interface.
            port (int): Port of the interface.

        Returns:
            bool: True if connected and false if not.
        """           
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.connect((ip, port))
            self.nanonis = nanonis_spm.Nanonis(connection)
            self.connected = True
        except:
            self.connected = False

        return self.connected
    
    def disconnect(self):
        """Disconnect from Nanonis.
        """        
        if self.connected:
            self.nanonis.close()
            self.connected = False
    
    def check_syntax(self, script: str):
        """Validate text script for correct commands and argument counts.

        Args:
            script (str): The text-based script.

        Returns:
            list[str]: A list of syntax errors (empty if valid). Checks:
                - Unknown commands
                - Wrong number of arguments
                - Invalid loop syntax
                - Unbalanced loops
        """       
        errors = []
        lines = [line.strip() for line in script.splitlines() if line.strip()]
        stack = []

        for line_num, line in enumerate(lines, start=1):
            if line.startswith("loop"):
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    errors.append(f"Line {line_num}: invalid loop syntax → {line}")
                else:
                    stack.append("loop")

            elif line == "end":
                if not stack:
                    errors.append(f"Line {line_num}: 'end' without matching 'loop' → {line}")
                else:
                    stack.pop()

            else:
                parts = line.split()
                cmd_name = parts[0]

                if cmd_name not in FUNCTION_REGISTRY:
                    errors.append(f"Line {line_num}: unknown command → {line}")
                    continue

                cmd_info = FUNCTION_REGISTRY[cmd_name]
                expected_user_args = sum(1 for a in cmd_info["args"].values() if a["user"])
                given_args = len(parts) - 1

                if given_args != expected_user_args:
                    errors.append(
                        f"Line {line_num}: wrong number of arguments for '{cmd_name}'. "
                        f"Expected {expected_user_args}, got {given_args} → {line}"
                    )

        if stack:
            errors.append("Unclosed 'loop' detected")

        return errors

    def parse_commands(self, script: str):
        """Parse a text-based script into a flat list of commands with arguments.

        The parser:
        - Splits the script into lines.
        - Handles nested loops ("loop X" ... "end").
        - Expands loops into repeated command blocks.
        - Validates syntax before parsing.

        Args:
            script (str): The raw text script with commands, one per line.

        Returns:
            tuple: (commands, errors)
                - commands (list[dict]): A flat list of parsed commands, where each command is a dict:
                                {
                                    "command": <str>,   # Command name as written in script
                                    "func": <callable>, # Function or method reference
                                    "args": <list>,     # Positional argument values
                                }
                - errors (list[str]): List of syntax error strings (empty if valid)
        """
        errors = self.check_syntax(script)
        if errors:
            return None, errors

        lines = [line.strip() for line in script.splitlines() if line.strip()]
        commands, _ = self._parse_block(lines)
        return commands, []
    
    def _parse_block(self, lines, index=0):
        """ Recursively parse a block of script lines, expanding loops into repeated commands.

        Args:
            lines (list[str]): List of non-empty script lines.
            index (int): Current index to start parsing from.

        Returns:
            tuple: (parsed_commands, next_index)
                - parsed_commands: list of dicts with keys "command", "func", "args". See also parse_commands for details.
                - next_index: next line index after this block (used for recursion)
        """

        """
        INSTANCE_MAP maps class names to pre-created instances.
        Used to automatically inject instances for class methods in FUNCTION_REGISTRY.
        """
        INSTANCE_MAP = {
            "Nanonis": self.nanonis,
            "ScriptingInterface": self,
        }
        result = []
        while index < len(lines):
            line = lines[index]

            if line.startswith("loop"):
                _, count_str = line.split()
                count = int(count_str)
                block, index = self._parse_block(lines, index + 1)
                result.extend(block * count)
                continue

            elif line == "end":
                return result, index + 1

            else:
                parts = line.split()
                cmd_name = parts[0]
                cmd_info = FUNCTION_REGISTRY[cmd_name]

                args = []
                user_values = parts[1:]
                user_index = 0
                for arg_name, arg_info in cmd_info["args"].items():
                    if arg_info["user"]:
                        args.append(arg_info["type"](user_values[user_index]))
                        user_index += 1
                    else:
                        args.append(arg_info["default"])

                func_ref = cmd_info["func"]
                if hasattr(func_ref, "__qualname__") and "." in func_ref.__qualname__:
                    cls_name = func_ref.__qualname__.split(".")[0]
                    instance = INSTANCE_MAP[cls_name]
                    func = func_ref.__get__(instance)
                else:
                    func = func_ref

                result.append({
                    "cmd": cmd_name,
                    "func": func,
                    "args": args,
                })
                index += 1

        return result, index
    
    def execute(self, script):
        """Execute a text-based script.

        Args:
            script (str): The text-based script.

        Returns:
            tuple: (errorString, response, variables)
                - errorString (str): Error messages.
                - response (str or bytes): Raw response from the called function(s).
                - variables (list): List of returned variables from the called function(s)
        """
        commands, errors = self.parse_commands(script)
        # print("PARSED:", commands)

        for entry in commands:
            cmd = entry["cmd"]
            func = entry["func"]
            args = entry["args"]
            errorString, response, variables = func(*args)

        return errorString, response, variables
   
    # -------------------------------
    # CUSTOM FUNCTIONS
    # -------------------------------
    def getXY(self):
        errorString, response, variables = self.nanonis.FolMe_XYPosGet(1)
        print(self.nanonis.FolMe_XYPosGet(1))
        x = variables[0]
        y = variables[1]
        return errorString, response, [x, y]
    
    def getZ(self):
        errorString, response, variables = self.nanonis.ZCtrl_ZPosGet()
        z = variables[0]
        return errorString, response, [z]

    def setXY(self, x, y):
        errorString, response, variables = self.nanonis.FolMe_XYPosSet(np.float64(x), np.float64(y), 1)
        return errorString, response, variables
    
    def setZ(self, z):
        errorString, response, variables = self.nanonis.ZCtrl_ZPosSet(np.float32(z))
        return errorString, response, variables

    def addX(self, dx):
        errorString, response, [x, y] = self.getXY()
        x = x + dx
        self.setXY(x, y)
        return errorString, response, [x, y]

    def addY(self, dy):
        errorString, response, [x, y] = self.getXY()
        y = y + dy
        self.setXY(x, y)
        return errorString, response, [x, y]
    
    def addZ(self, dz):
        errorString, response, [z] = self.getZ()
        z = z + dz
        self.setZ(z)
        return errorString, response, [z]
    
    def addBias(self, dV):
        errorString, response, variables = self.nanonis.Bias_Get()
        bias = variables[0]
        bias = bias + dV
        errorString, response, variables = self.nanonis.Bias_Set(np.float32(bias))
        return errorString, response, variables
    
    def addCurrent(self, dI):
        errorString, response, variables = self.nanonis.ZCtrl_SetpntGet()
        current = variables[0]
        current = current + dI
        error, response, variables = self.nanonis.ZCtrl_SetpntSet(np.float32(current))
        return errorString, response, variables

    def wait(self, dt):
        time.sleep(float(dt))
        errorString = ''
        response = ''
        variables = []
        return errorString, response, variables
        
    def correctZDrift(self, duration):
        errorString, response, variables = self.execute('drift.Get')
        comp_status = variables[0]
        old_vx = variables[1]
        old_vy = variables[2]
        old_vz = variables[3]

        dt = 1 
        n = int(np.float32(duration / dt))
        t_data = np.zeros(n)
        z_data = np.zeros(n)

        for i in range(len(t_data)):
            errorString, response, [z] = self.getZ()
            t_data[i] = dt * i
            z_data[i] = z
            time.sleep(int(dt))

        def lin_func(x, m, b):
            return x * m + b

        popt, _ = curve_fit(lin_func, t_data, z_data)
        vz = popt[0]

        if comp_status == 1:
            new_vz = old_vz + vz
        else:
            old_vx = 0
            old_vy = 0
            new_vz = vz
        errorString, response, variables = self.execute(f'drift.Set 1 {old_vx} {old_vy} {new_vz}')
        return errorString, response, variables


# -------------------------------
# FUNCTION_REGISTRY
# -------------------------------
"""
FUNCTION_REGISTRY is a dictionary mapping command names (strings) to function metadata.

Each entry defines:
    "func": reference to function or unbound class method
    "args": dictionary of arguments:
        - "type": Python (numpy) type for input conversion
        - "default": default value if not user-supplied
        - "user": True if user must provide argument, False if defaulted

Remote commands are added to FUNCTION_REGISTRY just like normal commands.

Example:
FUNCTION_REGISTRY = {
    "add": {
        "func": add,
        "args": {
            "a": {"type": int, "default": 0, "user": True},
            "b": {"type": int, "default": 2, "user": False},
        },
    },
}
"""
FUNCTION_REGISTRY = {
    "bias.Set": {
        "func": nanonis_spm.Nanonis.Bias_Set,
        "args": {
            "Bias value (V)": {"type": np.float32, "default": 0.5, "user": True}
        },
    },
    "bias.Get": {
        "func": nanonis_spm.Nanonis.Bias_Get,
        "args": {
        },
    },
    "current.Set": {
        "func": nanonis_spm.Nanonis.ZCtrl_SetpntSet,
        "args": {
            "Z-Controller setpoint": {"type": np.float32, "default": 100e-12, "user": True},
        },
    },
    "current.Get": {
        "func": nanonis_spm.Nanonis.ZCtrl_SetpntGet,
        "args": {          
        },
    },
    "fb.Set": {
        "func": nanonis_spm.Nanonis.ZCtrl_OnOffSet,
        "args": {
            "Z-Controller status": {"type": np.uint32, "default": 1, "user": True},
        },
    },
    "fb.Get": {
        "func": nanonis_spm.Nanonis.ZCtrl_OnOffGet,
        "args": {
        },
    },
    "biasSpec.Start": {
        "func": nanonis_spm.Nanonis.BiasSpectr_Start,
        "args": {
            "Get data": {"type": np.uint32, "default": 0, "user": False},
            "Save base name": {"type": str, "default": "", "user": False},
        },
    },
    "biasSpec.LimitsGet": {
        "func": nanonis_spm.Nanonis.BiasSpectr_LimitsGet,
        "args": {
        },
    },
    "biasSpec.LimitsSet": {
        "func": nanonis_spm.Nanonis.BiasSpectr_LimitsSet,
        "args": {
            "Start value (V)": {"type": np.float32, "default": 0.1, "user": True},
            "End value (V)": {"type": np.float32, "default": 1.0, "user": True},
        },
    },
    "scan.Start": {
        "func": nanonis_spm.Nanonis.Scan_Action,
        "args": {
            "Scan action": {"type": np.uint16, "default": 0, "user": False},
            "Scan direction": {"type": np.uint32, "default": 0, "user": False},
        },
    },
    "scan.Wait": {
        "func": nanonis_spm.Nanonis.Scan_WaitEndOfScan,
        "args": {
            "Timeout (ms)": {"type": np.int32, "default": -1, "user": False},
        },
    },
    "lockin.PhaseSet": {
        "func": nanonis_spm.Nanonis.LockIn_DemodPhasSet,
        "args": {
            "Demodulator number": {"type": np.int32, "default": 1, "user": False},
            "Phase (deg)": {"type": np.float32, "default": 0.0, "user": True},
        },
    },
    "lockin.PhaseGet": {
        "func": nanonis_spm.Nanonis.LockIn_DemodPhasGet,
        "args": {
            "Demodulator number": {"type": np.int32, "default": 1, "user": False},        
        },
    },
    "lockin.AmplSet": {
        "func": nanonis_spm.Nanonis.LockIn_ModAmpSet,
        "args": {
            "Modulator number": {"type": np.int32, "default": 1, "user": False},
            "Amplitude": {"type": np.float32, "default": 1e-3, "user": True},
        },
    },
    "lockin.AmplGet": {
        "func": nanonis_spm.Nanonis.LockIn_ModAmpGet,
        "args": {
            "Modulator number": {"type": np.int32, "default": 1, "user": False}, 
        },
    },
    "lockin.FreqSet": {
        "func": nanonis_spm.Nanonis.LockIn_ModPhasFreqSet,
        "args": {
            "Modulator number": {"type": np.int32, "default": 1, "user": False},
            "Frequency (Hz)": {"type": np.float32, "default": 187.0, "user": True},
        },
    },
    "lockin.FreqGet": {
        "func": nanonis_spm.Nanonis.LockIn_ModPhasFreqGet,
        "args": {
            "Modulator number": {"type": np.int32, "default": 1, "user": False}, 
        },
    },
    "atomtrack.ModSet": {
        "func": nanonis_spm.Nanonis.AtomTrack_CtrlSet,
        "args": {
            "AT control": {"type": np.uint16, "default": 0, "user": False},
            "Status": {"type": np.uint16, "default": 0, "user": True},
        },
    },
    "atomtrack.TrackSet": {
        "func": nanonis_spm.Nanonis.AtomTrack_CtrlSet,
        "args": {
             "AT control": {"type": np.uint16, "default": 1, "user": False},
            "Status": {"type": np.uint16, "default": 0, "user": True},
        },
    },
    "withdraw": {
        "func": nanonis_spm.Nanonis.ZCtrl_Withdraw,
        "args": {
            "Wait until finished": {"type": np.uint32, "default": 1, "user": False},
            "punTimeout (ms)ct": {"type": np.int32, "default": -1, "user": False},
        },
    },
    "drift.Get": {
        "func": nanonis_spm.Nanonis.Piezo_DriftCompGet,
        "args": {
        },
    },
    "drift.Set": {
        "func": nanonis_spm.Nanonis.Piezo_DriftCompSet,
        "args": {
            "Compensation status": {"type": np.uint32, "default": 1, "user": True},
            "Vx (m/s)": {"type": np.float32, "default": 0.0, "user": True},
            "Vy (m/s)": {"type": np.float32, "default": 0.0, "user": True},
            "Vz (m/s)": {"type": np.float32, "default": 0.0, "user": True},
            "Saturation limit (%)": {"type": np.float32, "default": 10.0, "user": False},
        },
    },
    "xy.Get": {
        "func": ScriptingInterface.getXY,
        "args": {
        },
    },
    "xy.Set": {
        "func": ScriptingInterface.setZ,
        "args": {
            "x (m)": {"type": np.float64, "default": 0.0, "user": True},
            "y (m)": {"type": np.float64, "default": 0.0, "user": True},
        },
    },
    "z.Get": {
        "func": ScriptingInterface.getZ,
        "args": {
        },
    },
    "z.Set": {
        "func": ScriptingInterface.setZ,
        "args": {
            "z (m)": {"type": np.float32, "default": 0.0, "user": True},
        },
    },
    "x.Add": {
        "func": ScriptingInterface.addX,
        "args": {
            "dx (m)": {"type": np.float64, "default": 0.0, "user": True},
        },
    },
    "y.Add": {
        "func": ScriptingInterface.addY,
        "args": {
            "dy (m)": {"type": np.float64, "default": 0.0, "user": True},
        },
    },
    "z.Add": {
        "func": ScriptingInterface.addZ,
        "args": {
            "dz (m)": {"type": np.float32, "default": 0.0, "user": True},
        },
    },
    "wait": {
        "func": ScriptingInterface.wait,
        "args": {
            "Time (s)": {"type": np.uint32, "default": 5, "user": True},
        },
    },
    "bias.Add": {
        "func": ScriptingInterface.addBias,
        "args": {
            "Bias value (V)": {"type": np.float32, "default": 0.1, "user": True},
        },
    },
    "current.Add": {
        "func": ScriptingInterface.addCurrent,
        "args": {
            "Current value (A)": {"type": np.float32, "default": 50e-12, "user": True},
        },
    },
    "drift.correctZ": {
        "func": ScriptingInterface.correctZDrift,
        "args": {
            "Time (s)": {"type": np.uint32, "default": 5, "user": True},
        },
    },
    "QuPe.FreqSet": {
        "func": lambda freq: TCPClient("QuPe").send("setFrequency", [freq]),
        "args": {
            "Frequency (Hz)": {"type": np.float32, "default": 1e9, "user": True},
        },
    },
}