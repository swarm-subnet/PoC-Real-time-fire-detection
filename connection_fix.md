# Tello Windows Connection Fix

This note documents a real failure we hit while provisioning RoboMaster TT / Tello drones from Windows.

## What Happened

The laptop could connect to the drone Wi-Fi (`TELLO-*`), but SDK commands failed immediately:

```text
OSError: [WinError 10051] A socket operation was attempted to an unreachable network
```

`ipconfig` showed:

```text
Wireless LAN adapter Wi-Fi:
  Autoconfiguration IPv4 Address: 169.254.x.x
  Subnet Mask: 255.255.0.0
  Default Gateway:
```

That means Windows associated with the Wi-Fi network, but did not have a valid IPv4 route to the Tello SDK address:

```text
Tello AP-mode IP: 192.168.10.1
SDK UDP port: 8889
```

The SDK was not failing. Python was not failing. The packet could not be sent because Windows had no route to `192.168.10.1`.

## Root Cause

We tried a static-IP workaround:

```powershell
netsh interface ipv4 set address name="Wi-Fi" static 192.168.10.2 255.255.255.0
```

This was the wrong default approach.

Why:

- It changes the global Windows Wi-Fi adapter IPv4 configuration.
- It is not scoped only to the current `TELLO-*` connection.
- If interrupted or only partially restored, Windows can keep stale IP objects/routes.
- After that, Windows may still show `169.254.x.x`, while also reporting:

```text
The object already exists.
```

when trying to apply `192.168.10.2` again.

In short: the static workaround can leave the Wi-Fi adapter in an inconsistent state.

## The Fix That Worked

Run these commands from **PowerShell as Administrator**:

```powershell
netsh wlan disconnect

Remove-NetIPAddress -InterfaceAlias "Wi-Fi" -IPAddress 192.168.10.2 -Confirm:$false -ErrorAction SilentlyContinue

Set-NetIPInterface -InterfaceAlias "Wi-Fi" -AddressFamily IPv4 -Dhcp Enabled
Set-DnsClientServerAddress -InterfaceAlias "Wi-Fi" -ResetServerAddresses

Get-NetRoute -AddressFamily IPv4 |
  Where-Object { $_.DestinationPrefix -eq "192.168.10.0/24" -or $_.DestinationPrefix -eq "192.168.10.1/32" } |
  Remove-NetRoute -Confirm:$false

netsh wlan show profiles
netsh wlan delete profile name="TELLO-XXXXXX"

Disable-NetAdapter -Name "Wi-Fi" -Confirm:$false
Start-Sleep 3
Enable-NetAdapter -Name "Wi-Fi" -Confirm:$false
```

What this does:

- Disconnects Wi-Fi.
- Removes stale `192.168.10.x` IP objects.
- Restores Wi-Fi IPv4 to DHCP.
- Restores DNS to DHCP.
- Removes stale `192.168.10.0/24` routes.
- Deletes saved `TELLO-*` Wi-Fi profiles.
- Restarts the Wi-Fi adapter.

After repair:

1. Connect to the target router Wi-Fi.
2. Run `ipconfig`.
3. Confirm Wi-Fi gets a router IP, e.g. `192.168.100.x`.
4. Connect manually to `TELLO-*` only if doing one-drone manual setup.

## Correct Mental Model

Connecting to `TELLO-*` has two separate layers:

1. Wi-Fi association succeeds.
2. Windows must have an IPv4 route to `192.168.10.1`.

Being connected to the SSID is not enough.

Healthy AP-mode state:

```text
PC Wi-Fi IP: 192.168.10.x
Drone IP:    192.168.10.1
SDK port:    UDP 8889
```

Broken AP-mode state:

```text
PC Wi-Fi IP: 169.254.x.x
Drone IP:    192.168.10.1
Result:      WinError 10051 / unreachable network
```

## Rules For Future Code

Do not change the Windows Wi-Fi adapter to static IP from Python.

Do not run this automatically:

```powershell
netsh interface ipv4 set address name="Wi-Fi" static 192.168.10.2 255.255.255.0
```

If Windows has no route to `192.168.10.1`, scripts should:

- Detect the problem.
- Explain that Windows DHCP/profile state is broken.
- Exit safely.
- Tell the user to restore DHCP/profile state manually from Administrator PowerShell.

## Useful Commands

Inspect Wi-Fi state:

```powershell
ipconfig
Get-NetIPConfiguration -InterfaceAlias "Wi-Fi"
Get-NetIPAddress -InterfaceAlias "Wi-Fi" -AddressFamily IPv4
Get-NetIPInterface -InterfaceAlias "Wi-Fi" -AddressFamily IPv4
route print 192.168.10.1
```

Restore DHCP manually:

```powershell
netsh interface ipv4 set address name="Wi-Fi" source=dhcp
netsh interface ipv4 set dnsservers name="Wi-Fi" source=dhcp
```

Restart Wi-Fi adapter:

```powershell
Disable-NetAdapter -Name "Wi-Fi" -Confirm:$false
Start-Sleep 3
Enable-NetAdapter -Name "Wi-Fi" -Confirm:$false
```

Delete saved Tello profiles:

```powershell
netsh wlan show profiles
netsh wlan delete profile name="TELLO-XXXXXX"
```

## Related Reference

See also:

- `tello.md` for RoboMaster TT / Tello SDK command and networking reference.
- `scripts/swarm/09_setup_new_drone.py` for one-drone manual station-mode setup.
- `scripts/swarm/14_auto_setup_tello_wifi.py` for serial station-mode setup.
