# RoboMaster TT / Tello SDK Reference

This file is a local developer reference for coding against the RoboMaster TT / Tello SDK 3.0.

Source context: SDK 3.0 User Guide, V1.0, 2021.04.

## Core Architecture

The drone is controlled over Wi-Fi using UDP text commands.

Default AP-mode network:

```text
Drone IP: 192.168.10.1
SDK command UDP port: 8889
State UDP port: 8890
Video UDP port: 11111
```

Basic command flow:

1. Connect PC to the drone Wi-Fi network, usually `TELLO-*` or `RMTT-*`.
2. Open a UDP client socket on the PC.
3. Send `command` to `192.168.10.1:8889`.
4. Wait for `ok`.
5. Send SDK commands to the same IP/port.

Important: `command` must be sent before any other SDK command.

## Safety Behavior

If the drone receives no command input for 15 seconds after executing the current command, it automatically lands.

Exception: the open-source controller may send `[TELLO] battery?` for battery status.

Practical implication for code:

- Keep sending periodic safe commands or telemetry queries during active flight.
- Do not leave the drone airborne without an active control loop.
- On process shutdown, send `land` if the drone may be airborne.
- For urgent aborts, send `emergency`, but note this stops motors immediately.

## Resetting Wi-Fi

To reset Wi-Fi settings:

1. Power on the drone.
2. Long-press the power button for 5 seconds.
3. The drone reboots after the status indicator goes out.
4. When the indicator quickly flashes yellow, SSID/password are reset to factory defaults.
5. Default Wi-Fi has no password.

## UDP Channels

### Commands And Responses

```text
PC/Mac/Mobile -> Tello
Tello IP: 192.168.10.1
UDP port: 8889
```

Responses:

- Control commands return `ok`, `error`, or a result/status code.
- Setting commands return `ok`, `error`, or a result/status code.
- Read commands return the requested value.

### State

```text
Tello -> PC/Mac/Mobile
Listen on: 0.0.0.0:8890
```

Set up a UDP server on the PC to receive state messages.

### Video

```text
Tello -> PC/Mac/Mobile
Listen on: 0.0.0.0:11111
```

To receive video:

1. Send `command`.
2. Send `streamon`.
3. Read UDP video packets on port `11111`.
4. Send `streamoff` to stop video.

## Command Types

### Control Commands

| Command | Description | Notes |
| --- | --- | --- |
| `command` | Enter SDK command mode | Must be first command |
| `takeoff` | Auto takeoff | Returns `ok` / `error` |
| `land` | Auto land | Returns `ok` / `error` |
| `streamon` | Turn video stream on | Video on UDP `11111` |
| `streamoff` | Turn video stream off | |
| `emergency` | Stop motors immediately | Use only for urgent abort |
| `stop` | Stop moving and hover | |
| `reboot` | Reboot drone | No response on success |
| `motoron` | Low-speed motor-on mode | Only when static standby |
| `motoroff` | Exit motor-on mode | |
| `throwfly` | Throw launch mode | Throw horizontally within 5s |

### Movement Commands

Distances are centimeters.

| Command | Description | Range |
| --- | --- | --- |
| `up x` | Fly upward | `x = 20..500` |
| `down x` | Fly downward | `x = 20..500` |
| `left x` | Fly left | `x = 20..500` |
| `right x` | Fly right | `x = 20..500` |
| `forward x` | Fly forward | `x = 20..500` |
| `back x` | Fly backward | `x = 20..500` |
| `cw x` | Rotate clockwise | `x = 1..360` degrees |
| `ccw x` | Rotate counterclockwise | `x = 1..360` degrees |
| `flip x` | Flip | `l`, `r`, `f`, `b` |

### Coordinate Commands

```text
go x y z speed
```

Fly to coordinates at speed:

- `x`: `-500..500`
- `y`: `-500..500`
- `z`: `-500..500`
- `speed`: `10..100` cm/s
- `x`, `y`, and `z` cannot all be between `-20` and `20`.

```text
curve x1 y1 z1 x2 y2 z2 speed
```

Fly in a curve:

- `x1`, `x2`: `-500..500`
- `y1`, `y2`: `-500..500`
- `z1`, `z2`: `-500..500`
- `speed`: `10..60` cm/s
- Curve radius must be between `0.5m` and `10m`.
- `x`, `y`, and `z` cannot all be between `-20` and `20`.

## Setting Commands

| Command | Description | Notes |
| --- | --- | --- |
| `speed x` | Set speed | `x = 10..100` cm/s |
| `rc a b c d` | Set remote-control channel values | No response |
| `wifi ssid pass` | Change drone Wi-Fi SSID/password | Reboots in 3s |
| `ap ssid pass` | Station mode: connect drone to router/AP | Reboots in 3s |
| `mon` | Enable mission pad detection | Downward by default |
| `moff` | Disable mission pad detection | |
| `mdirection x` | Mission pad detection direction | `0` down, `1` forward, `2` both |
| `port info video` | Set state/video ports | Ports `1025..65535` |
| `setfps fps` | Set video FPS | `high`, `middle`, `low` = 30, 15, 5 FPS |
| `setbitrate bitrate` | Set video bitrate | `0..5`, where `0` is auto |
| `setresolution resolution` | Set video resolution | `high` = 720p, `low` = 480p |

RC channel values:

```text
rc a b c d
a = roll      -100..100
b = pitch     -100..100
c = throttle  -100..100
d = yaw       -100..100
```

## Read Commands

| Command | Description | Response |
| --- | --- | --- |
| `speed?` | Current speed | `10..100` |
| `battery?` | Battery percentage | `0..100` usually |
| `time?` | Motor running time | seconds |
| `wifi?` | Wi-Fi SNR | SNR |
| `sdk?` | SDK version | version number |
| `sn?` | Serial number | production SN |
| `hardware?` | Hardware type | `TELLO` / `RMTT` |
| `wifiversion?` | Open-source controller Wi-Fi version | controller only |
| `ap?` | Current router SSID/password target | controller only |
| `ssid?` | Current drone SSID | controller only |

## Station Mode

To connect the drone to a router/AP:

```text
command
ap <ssid> <password>
```

Expected response:

```text
OK, drone will reboot in 3s
```

After reboot, the drone no longer uses `192.168.10.1`. It gets an IP from the router DHCP server.

Practical flow:

1. Connect PC to `TELLO-*`.
2. Send `command`.
3. Send `ap <router_ssid> <router_password>`.
4. Wait for drone reboot.
5. Reconnect PC to the router Wi-Fi.
6. Scan the router subnet for drones by sending `command` to UDP port `8889`.

## Motor-On Mode

`motoron` enters low-speed propeller rotation mode.

Purpose:

- Indicates TT is ready for takeoff.
- Helps heat dissipation.
- Can avoid shutdown caused by excessive temperature.

Restriction:

- Only use while the drone is in static standby.
- After takeoff, the drone automatically exits Motor-On mode.

## Bench Thermal Policy

The SDK exposes board temperature in state packets as `templ` and `temph`, but
does not publish a precise shutdown threshold. For bench dashboards, use
conservative thresholds and hysteresis:

- Warning: `temph >= 75C`
- Motor-on cooling start: `temph >= 78C`
- Video auto-stop: `temph >= 82C`
- Motor-on cooling stop: `temph <= 75C`, after at least 45 seconds of cooling

`motoron` cooling must only be used when drones are stationary on the bench and
the prop area is clear. Do not use it as an airborne cooling mechanism.

## Mission Pad Commands

Mission pad commands require mission pads and detection enabled.

Enable:

```text
mon
```

Detection direction:

```text
mdirection 0  # downward
mdirection 1  # forward
mdirection 2  # both
```

Frequency:

- Downward only: 20 Hz
- Forward only: 20 Hz
- Both: alternating, 10 Hz each

Commands using mission pads:

```text
go x y z speed mid
curve x1 y1 z1 x2 y2 z2 speed mid
jump x y z speed yaw mid1 mid2
```

Mission pad IDs:

- `m1..m8`: specific mission pad ID
- `m-1`: first pad identified internally
- `m-2`: nearest pad

## Open-Source Controller / ESP32

Commands from ESP32 to Tello use:

```text
[TELLO] <command>
```

Tello responses to ESP32 use:

```text
ETT <response>\r\n
```

Example:

```text
ESP32 -> Tello: [TELLO] takeoff
Tello -> ESP32: ETT ok\r\n
```

## ESP32 Extension Commands

These apply to the TT open-source controller.

### LED

```text
EXT led r g b
EXT led br t r g b
EXT led bl t r1 g1 b1 r2 g2 b2
```

### Matrix LED

```text
EXT mled g xxxx
EXT mled l/r/u/d r/b/p t xxxx
EXT mled s r/b/p xxxx
EXT mled sg xxxx
EXT mled sc
EXT mled sl n
```

Matrix pattern characters:

- `r`: red
- `b`: blue
- `p`: purple
- `0`: off

Max pattern length: 64.

### ToF

```text
EXT tof?
```

Response:

```text
tof xxxx
```

Unit: millimeters.

`8192` means detection range exceeded.

### ESP32 Version

```text
EXT version?
```

Response:

```text
esp32vx.x.x.x
```

## Tello State Format

State is a string sent over UDP port `8890`.

Example shape:

```text
mid:%d;x:%d;y:%d;z:%d;mpry:%d,%d,%d;pitch:%d;roll:%d;yaw:%d;
vgx:%d;vgy:%d;vgz:%d;templ:%d;temph:%d;tof:%d;h:%d;bat:%d;
baro:%f;time:%d;agx:%f;agy:%f;agz:%f;
```

Common fields:

| Field | Meaning | Unit |
| --- | --- | --- |
| `mid` | Mission pad ID | ID |
| `x` | X position relative to mission pad | cm |
| `y` | Y position relative to mission pad | cm |
| `z` | Z position relative to mission pad | cm |
| `mpry` | Mission pad pitch/yaw/roll | degrees |
| `pitch` | Drone pitch | degrees |
| `roll` | Drone roll | degrees |
| `yaw` | Drone yaw | degrees |
| `vgx` | X speed | dm/s |
| `vgy` | Y speed | dm/s |
| `vgz` | Z speed | dm/s |
| `templ` | Minimum board temperature | Celsius |
| `temph` | Maximum board temperature | Celsius |
| `tof` | ToF distance | cm |
| `h` | Height from takeoff point | cm |
| `bat` | Battery percentage | percent |
| `baro` | Barometer height | m |
| `time` | Motor running time | s |
| `agx` | X acceleration | cm/s2 |
| `agy` | Y acceleration | cm/s2 |
| `agz` | Z acceleration | cm/s2 |

Mission pad special values:

- `mid = -2`: mission pad detection not enabled.
- `mid = -1`: detection enabled, but no pad detected.
- `x/y/z = -200`: detection not enabled.
- `x/y/z = -100`: detection enabled, but no pad detected.

## Coding Notes For This Repo

Prefer this SDK sequence:

```text
command
battery?
streamoff
streamon
...
land
streamoff
```

For safe flight scripts:

- Always query battery before takeoff.
- Keep emergency/land path available.
- Use conservative movement distances.
- Send `land` on `q`, Ctrl+C, or exceptions if airborne.
- Avoid `emergency` except when an immediate motor stop is intended.
- Avoid commands under the minimum motion distance; many movement commands require at least 20 cm.
- For yaw correction, prefer small values like `cw 10` / `ccw 10` when close to center.

For networking:

- AP mode: drone is `192.168.10.1`.
- Station mode: scan router subnet for the drone IP.
- Commands use UDP `8889`.
- State uses UDP `8890`.
- Video uses UDP `11111`.
- If Windows connects to `TELLO-*` but gets `169.254.x.x`, DHCP failed locally; SDK commands to `192.168.10.1` will likely fail until the adapter has a `192.168.10.x` route.
