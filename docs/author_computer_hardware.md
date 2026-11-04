# Specifications of the author's computer

The first author ran r07, r08 and r12 and the re-runs of 31 October 2026 (`results/reproduction/`) on his own
computer, which thesis Table 3.5 describes. He recorded its specifications on 4 November 2026 at 19:44 (UTC+3) with
the PowerShell command below, which reads only the manufacturer, model, processor, installed memory, graphics,
drives and Windows version (no computer name, user name or serial number). The values are transcribed from the
output he shared.

| Item | Value |
| --- | --- |
| Manufacturer | HP |
| Model | HP Laptop 15-fd0xxx |
| Processor | 13th Gen Intel(R) Core(TM) i3-1315U |
| Installed memory | 8 GB |
| Graphics | Intel(R) UHD Graphics (no other graphics adapter) |
| Drive | WD PC SN740 SDDPNQD-256G-2006, 238 GB as reported by Windows (256 GB nominal) |
| Windows | Microsoft Windows 11 Home 64-bit (build 26200) |

The cloud workspace used for the other runs is described in `docs/REPRODUCIBILITY.md`; its hardware was not
recorded.

Command (Windows PowerShell):

```powershell
$cs=Get-CimInstance Win32_ComputerSystem; $os=Get-CimInstance Win32_OperatingSystem; $o=[ordered]@{}; $o.Manufacturer=$cs.Manufacturer; $o.Model=$cs.Model; $o.Processor=((Get-CimInstance Win32_Processor).Name -join '; '); $o.InstalledMemoryGB=[math]::Round(((Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum)/1GB,1); $o.Graphics=((Get-CimInstance Win32_VideoController).Name -join '; '); $o.Drives=((Get-CimInstance Win32_DiskDrive | ForEach-Object { '{0} {1} GB' -f $_.Model,[math]::Round($_.Size/1GB) }) -join '; '); $o.Windows=('{0} {1} (build {2})' -f $os.Caption,$os.OSArchitecture,$os.BuildNumber); $o.Recorded=(Get-Date -Format 'yyyy-MM-dd HH:mm'); $f=Join-Path ([Environment]::GetFolderPath('Desktop')) 'hardware_info.txt'; [pscustomobject]$o | Format-List | Out-File $f -Encoding utf8; Get-Content $f
```
