from typing import Any


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator


ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
ACTIONS_DESCRIPTION = {
    0: 'change lane to the left',
    1: 'remain in current lane at current speed',
    2: 'change lane to the right',
    3: 'accelerate',
    4: 'decelerate',
}


class getAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(
        name='Get Available Actions',
        description="Use this first before deciding. Returns available actions for the ego car. Input: 'ego'."
    )
    def inference(self, input: str) -> str:
        available = self.env.get_available_actions()
        lines = ['You can ONLY use one of the following actions:']
        for a in available:
            lines.append(f"  {ACTIONS_ALL[a]} -- {ACTIONS_DESCRIPTION[a]}")
        if 1 in available:
            lines.append('Check IDLE first.')
        if 0 in available or 2 in available:
            lines.append('For lane changes, carefully check target lane safety.')
        if 3 in available:
            lines.append('Consider acceleration carefully.')
        if 4 in available:
            lines.append('Deceleration is LAST priority.')
        lines.append("""
Safety check steps:
  Step 1: Identify lanes affected (accel/decel/idle → current lane; lane change → target lane).
  Step 2: Get vehicles in that lane.
  Step 3: Check safety with each vehicle one by one using the appropriate tool.
""")
        return '\n'.join(lines)


class getAvailableLanes:
    def __init__(self, sce) -> None:
        self.sce = sce

    @prompts(
        name='Get Available Lanes',
        description="Returns available lanes for a vehicle. Input: vehicle id string (e.g. 'ego')."
    )
    def inference(self, vid: str) -> str:
        veh = self.sce.vehicles[vid]
        idx = self.sce.lanes[veh.lane_id].laneIdx
        cur = veh.lane_id
        parts = [f"`{cur}` is the current lane."]
        if idx > 0:
            left = f'lane_{idx - 1}'
            parts.append(f"`{left}` is to the left.")
        if idx < 3:
            right = f'lane_{idx + 1}'
            parts.append(f"`{right}` is to the right.")
        return ' '.join(parts)


class getLaneInvolvedCar:
    def __init__(self, sce) -> None:
        self.sce = sce

    @prompts(
        name='Get Lane Involved Car',
        description="Returns cars that may affect your action in a specific lane. Input: lane id (e.g. 'lane_1'). Call Get Available Lanes first."
    )
    def inference(self, laneID: str) -> str:
        if laneID not in {'lane_0', 'lane_1', 'lane_2', 'lane_3'}:
            return "Invalid lane id. Use Get Available Lanes first."
        ego = self.sce.vehicles['ego']
        lane_vehs = sorted(
            [(v.id, v.lanePosition) for k, v in self.sce.vehicles.items()
             if k != 'ego' and v.lane_id == laneID],
            key=lambda x: x[1]
        )
        if not lane_vehs:
            return f"No cars on {laneID}. Lane is safe."

        leading_idx = next((i for i, (_, pos) in enumerate(lane_vehs) if pos >= ego.lanePosition), -1)

        if leading_idx == -1:
            return f"{lane_vehs[-1][0]} is behind ego on {laneID}. Check conflict."
        elif leading_idx == 0:
            vid, pos = lane_vehs[0]
            spd = round(self.sce.vehicles[vid].speed, 1)
            dist = round(pos - ego.lanePosition, 2)
            return f"{vid} is ahead of ego on {laneID} at {spd}m/s, {dist}m away. Check conflict."
        else:
            lead_id, lead_pos = lane_vehs[leading_idx]
            rear_id = lane_vehs[leading_idx - 1][0]
            spd = round(self.sce.vehicles[lead_id].speed, 1)
            dist = round(lead_pos - ego.lanePosition, 2)
            return f"{lead_id} is ahead ({spd}m/s, {dist}m) and {rear_id} is behind ego on {laneID}. Check conflict with both."


class isChangeLaneConflictWithCar:
    TIME_HEAD_WAY = 3.0
    VEHICLE_LENGTH = 5.0

    def __init__(self, sce) -> None:
        self.sce = sce

    @prompts(
        name='Is Change Lane Conflict With Car',
        description="Check if a lane change conflicts with a specific car. Input: 'lane_id, vehicle_id' (comma-separated)."
    )
    def inference(self, inputs: str) -> str:
        laneID, vid = inputs.replace(' ', '').split(',')
        if vid not in self.sce.vehicles:
            return "Invalid vehicle id. Use Get Lane Involved Car first."
        ego = self.sce.vehicles['ego']
        veh = self.sce.vehicles[vid]
        if veh.lanePosition >= ego.lanePosition:
            rel_spd = ego.speed - veh.speed
            safe = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * rel_spd
        else:
            rel_spd = veh.speed - ego.speed
            safe = ego.lanePosition - veh.lanePosition - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * rel_spd
        if safe:
            return f"Lane change to `{laneID}` is safe with `{vid}`."
        return f"Lane change to `{laneID}` may conflict with `{vid}`. Unsafe."


class isAccelerationConflictWithCar:
    TIME_HEAD_WAY = 5.0
    VEHICLE_LENGTH = 5.0
    ACCELERATION = 4.0

    def __init__(self, sce) -> None:
        self.sce = sce

    @prompts(
        name='Is Acceleration Conflict With Car',
        description="Check if accelerating is safe with a specific car. Input: vehicle id string."
    )
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles or vid == 'ego':
            return "Invalid vehicle id. Use Get Lane Involved Car first."
        ego = self.sce.vehicles['ego']
        veh = self.sce.vehicles[vid]
        if veh.lane_id != ego.lane_id:
            return f"{vid} is not in ego's lane. Check lane first."
        if veh.lanePosition >= ego.lanePosition:
            rel_spd = ego.speed + self.ACCELERATION - veh.speed
            dist = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH * 2
            if dist > self.TIME_HEAD_WAY * rel_spd:
                return f"Acceleration is safe with `{vid}`."
            return f"Acceleration may conflict with `{vid}`. Unsafe."
        return f"Acceleration is safe with `{vid}` (it's behind ego)."


class isKeepSpeedConflictWithCar:
    TIME_HEAD_WAY = 5.0
    VEHICLE_LENGTH = 5.0

    def __init__(self, sce) -> None:
        self.sce = sce

    @prompts(
        name='Is Keep Speed Conflict With Car',
        description="Check if keeping current speed is safe with a specific car. Input: vehicle id string."
    )
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles or vid == 'ego':
            return "Invalid vehicle id. Use Get Lane Involved Car first."
        ego = self.sce.vehicles['ego']
        veh = self.sce.vehicles[vid]
        if veh.lane_id != ego.lane_id:
            return f"{vid} is not in ego's lane."
        if veh.lanePosition >= ego.lanePosition:
            rel_spd = ego.speed - veh.speed
            dist = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH * 2
            if dist > self.TIME_HEAD_WAY * rel_spd:
                return f"Keeping speed is safe with `{vid}`."
            return f"Keeping speed may conflict with `{vid}`. Consider decelerating."
        return f"Keeping speed is safe with `{vid}` (it's behind ego)."


class isDecelerationSafe:
    TIME_HEAD_WAY = 3.0
    VEHICLE_LENGTH = 5.0
    DECELERATION = 3.0

    def __init__(self, sce) -> None:
        self.sce = sce

    @prompts(
        name='Is Deceleration Safe',
        description="Check if decelerating is safe with a specific car. Input: vehicle id string."
    )
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles or vid == 'ego':
            return "Invalid vehicle id. Use Get Lane Involved Car first."
        ego = self.sce.vehicles['ego']
        veh = self.sce.vehicles[vid]
        if veh.lane_id != ego.lane_id:
            return f"{vid} is not in ego's lane."
        if veh.lanePosition >= ego.lanePosition:
            rel_spd = ego.speed - veh.speed - self.DECELERATION
            dist = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH
            if dist > self.TIME_HEAD_WAY * rel_spd:
                return f"Deceleration is safe with `{vid}`."
            return f"Deceleration may conflict with `{vid}`. Slow down as much as possible."
        return f"Deceleration is safe with `{vid}` (it's behind ego)."