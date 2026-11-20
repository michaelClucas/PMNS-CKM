# Pin every process started from this repo (by executable path or command line) to this program's CPU set.
# Run before dispatching subagents and after they return. Prints each process it had to fix.
$mask = 0x1F0
$root = (Resolve-Path "$PSScriptRoot\..").Path
$fwd = '/' + $root.Substring(0,1).ToLower() + ($root.Substring(2).Replace('\','/'))
Get-CimInstance Win32_Process | Where-Object {
    ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, 'OrdinalIgnoreCase')) -or
    ($_.CommandLine -and ($_.CommandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                          $_.CommandLine.IndexOf($fwd, [StringComparison]::OrdinalIgnoreCase) -ge 0))
} | ForEach-Object {
    try {
        $p = Get-Process -Id $_.ProcessId -ErrorAction Stop
        if ([int64]$p.ProcessorAffinity -ne $mask) {
            $p.ProcessorAffinity = [IntPtr]$mask
            "pinned $($_.ProcessId) $($_.Name)"
        }
    } catch {}
}
