<#
.SYNOPSIS
    Checks for Linux/Unix commands in files and suggests PowerShell alternatives.

.DESCRIPTION
    This script scans files for common Linux/Unix commands and suggests PowerShell equivalents.
    It helps ensure cross-platform compatibility in scripts and documentation.

.PARAMETER Path
    The directory path to scan. Defaults to current directory.

.PARAMETER FileTypes
    Array of file extensions to check. Defaults to common script and markdown files.

.EXAMPLE
    .\Check-WindowsCompliance.ps1 -Path "C:\MyProject"
    
    Scans all script files in the specified directory for Linux/Unix commands.
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateScript({
        if (Test-Path -Path $_) { $true }
        else { throw "Path '$_' does not exist" }
    })]
    [string]$Path = ".",
    
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string[]]$FileTypes = @('.ps1', '.psm1', '.bat', '.cmd', '.md')
)

# Command mappings
# Common Linux/Unix to PowerShell command mappings
$script:commandMappings = @{
    # Command chaining and flow control
    '&&' = ';'  # Use ; for command chaining in PowerShell
    '`' = ';'   # Line continuation in PowerShell
    '| xargs' = '| ForEach-Object'  # Pipeline processing
    
    # File and directory operations
    'ls' = 'Get-ChildItem'
    'cat' = 'Get-Content'
    'rm ' = 'Remove-Item -Force '  # Added -Force for non-interactive use
    'rm -rf' = 'Remove-Item -Recurse -Force '  # Common recursive force delete
    'rmdir' = 'Remove-Item -Recurse -Force '  # Directory removal
    'mkdir -p' = 'New-Item -ItemType Directory -Force -Path '  # Create directory tree
    'cp ' = 'Copy-Item -Force '  # File copy with force
    'cp -r' = 'Copy-Item -Recurse -Force '  # Recursive copy
    'mv ' = 'Move-Item -Force '  # Move/rename with force
    'chmod' = 'Set-Acl'  # Or icacls for basic permissions
    'chown' = 'Set-Acl'  # Ownership changes
    
    # Text processing
    'grep' = 'Select-String'
    'sed' = 'ForEach-Object { $_ -replace }'  # Basic sed replacement pattern
    'awk' = 'Select-String -Pattern | ForEach-Object { $_.Matches.Groups[1].Value }'  # Basic awk pattern
    
    # System info
    'pwd' = 'Get-Location'
    'which' = 'Get-Command -Name'  # Find command location
    'uname' = '$PSVersionTable.OS'  # OS information
    
    # Output
    'echo' = 'Write-Output'  # Or just output directly
    'printf' = 'Write-Output'  # Basic printf functionality
    
    # Process management
    'ps' = 'Get-Process'
    'kill' = 'Stop-Process -Id'  # Or Stop-Process -Name
    'killall' = 'Get-Process -Name | Stop-Process'  # Kill by process name
    
    # Network
    'ifconfig' = 'Get-NetIPConfiguration'
    'netstat' = 'Get-NetTCPConnection'
    'ping' = 'Test-Connection'
    'wget' = 'Invoke-WebRequest -OutFile'  # Or use curl alias
    'curl' = 'Invoke-WebRequest'  # Or use curl alias
    
    # Package management
    'apt-get' = 'Install-Package'  # Or choco/chocolatey
    'yum' = 'Install-Package'  # Or choco/chocolatey
    'brew' = 'choco install'  # For Windows
    
    # Environment
    'export' = '$env:'  # Set environment variable
    'source' = '.'  # Dot sourcing in PowerShell
    
    # User management
    'sudo' = 'Start-Process -Verb RunAs'  # Admin elevation
    'su' = 'Start-Process powershell -Verb RunAs'  # Switch user/run as admin
    'whoami' = '$env:USERNAME'  # Or [System.Environment]::UserName
    
    # File permissions (Windows ACL)
    'chmod +x' = 'icacls "file" /grant "$env:USERNAME:(RX)"'  # Make executable
    'chmod 755' = 'icacls "file" /grant:r "$env:USERNAME:(OI)(CI)F"'  # Common permission set
    
    # Compression/Archives
    'tar -xzf' = 'Expand-Archive -Path archive.tar.gz -DestinationPath .'  # Extract .tar.gz
    'tar -czf' = 'Compress-Archive -Path source -DestinationPath archive.zip'  # Create .zip
    'unzip' = 'Expand-Archive -Path file.zip -DestinationPath .'  # Extract .zip
    
    # Text processing advanced
    'head' = 'Select-Object -First 10'  # First 10 lines
    'tail' = 'Select-Object -Last 10'  # Last 10 lines
    'wc -l' = 'Measure-Object -Line | Select-Object -ExpandProperty Lines'  # Line count
    'sort' = 'Sort-Object'  # Sort lines
    'uniq' = 'Get-Unique'  # Get unique lines
    
    # System monitoring
    'top' = 'Get-Process | Sort-Object -Property CPU -Descending | Select-Object -First 10'
    'df' = 'Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free, @{Name="UsedGB";Expression={$_.Used/1GB -as [int]}}, @{Name="FreeGB";Expression={$_.Free/1GB -as [int]}}'
    'du' = 'Get-ChildItem -Recurse | Measure-Object -Property Length -Sum | Select-Object @{Name="SizeGB";Expression={$_.Sum/1GB}}'
    
    # Date and time
    'date' = 'Get-Date -Format "yyyy-MM-dd HH:mm:ss"'  # Current date/time
    
    # File searching
    'find' = 'Get-ChildItem -Recurse -Filter'  # Basic file search
    'locate' = 'Get-ChildItem -Recurse -Include'  # File search with pattern
    
    # Network diagnostics
    'traceroute' = 'Test-NetConnection -TraceRoute'  # Network path tracing
    'nslookup' = 'Resolve-DnsName'  # DNS lookup
    'dig' = 'Resolve-DnsName'  # DNS lookup (alternative)
    
    # Process monitoring
    'pgrep' = 'Get-Process | Where-Object { $_.ProcessName -like "*pattern*" } | Select-Object Id, ProcessName'
    'pkill' = 'Get-Process | Where-Object { $_.ProcessName -like "*pattern*" } | Stop-Process -Force'
    
    # System information
    'free' = 'Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory, TotalVirtualMemorySize, FreeVirtualMemory'
    'lscpu' = 'Get-CimInstance -ClassName Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed'
    'lsblk' = 'Get-Disk | Get-Partition | Get-Volume | Select-Object DriveLetter, FileSystemLabel, Size, SizeRemaining'
    
    # User and session management
    'who' = 'Get-Process -IncludeUserName | Select-Object ProcessName, UserName -Unique'
    'w' = 'Get-Process -IncludeUserName | Select-Object ProcessName, UserName, CPU -First 10'
    'last' = 'Get-EventLog -LogName Security -InstanceId 4624, 4625 -After (Get-Date).AddDays(-7) | Select-Object TimeGenerated, Message'
    
    # File transfer
    'scp' = 'Copy-Item -Path "source" -Destination "user@server:path" -ToSession (New-PSSession -ComputerName server)'
    'rsync' = 'Robocopy'  # Or use third-party rsync for Windows
    
    # System services
    'systemctl' = 'Get-Service | Where-Object { $_.Status -eq "Running" }'  # List running services
    'service' = 'Get-Service'  # Service management
    
    # Package management (Windows specific)
    'choco' = 'choco'  # Chocolatey package manager
    'scoop' = 'scoop'  # Scoop package manager
    'winget' = 'winget'  # Windows Package Manager
    
    # Development tools
    'make' = 'nmake'  # Or use WSL for full make functionality
    'gcc' = 'cl.exe'  # MSVC compiler
    'g++' = 'cl.exe'  # MSVC C++ compiler
    
    # Version control
    'git' = 'git'  # Git is available cross-platform
    'svn' = 'svn'  # Subversion
    'hg' = 'hg'    # Mercurial
    
    # Text editors
    'nano' = 'notepad'  # Basic text editing
    'vim' = 'vim'  # If Vim is installed
    'emacs' = 'emacs'  # If Emacs is installed
    
    # Shell and environment
    'alias' = 'Set-Alias'  # Or use function for more complex aliases
    'history' = 'Get-History'  # Command history
    'type' = 'Get-Content'  # View file contents
    'touch' = 'New-Item -ItemType File -Force'  # Create empty file
    
    # File comparison
    'diff' = 'Compare-Object'  # Or use git diff if available
    'cmp' = 'Compare-Object'  # File comparison
    
    # File permissions (Windows specific)
    'stat' = 'Get-ItemProperty'  # File status
    'file' = 'Get-Item'  # File type information
    
    # System information (advanced)
    'lspci' = 'Get-PnpDevice | Where-Object { $_.Class -eq "System devices" } | Select-Object Name, Status'
    'lsusb' = 'Get-PnpDevice | Where-Object { $_.Class -eq "USB" } | Select-Object Name, Status'
    
    # Process management (advanced)
    'nohup' = 'Start-Process -NoNewWindow -PassThru'  # Run process in background
    'bg' = 'Start-Job'  # Background jobs
    'fg' = 'Receive-Job -Wait -AutoRemoveJob'  # Foreground jobs
    
    # Network (advanced)
    'ssh' = 'ssh'  # Use native Windows OpenSSH
    'sftp' = 'sftp'  # SFTP client
    'rsync' = 'robocopy'  # Alternative for rsync
    
    # System administration
    'crontab' = 'Get-ScheduledTask | Where-Object { $_.State -eq "Ready" }'  # Scheduled tasks
    'at' = 'New-ScheduledTaskAction'  # Schedule tasks
    
    # File system
    'mount' = 'Get-Volume'  # List mounted volumes
    'umount' = 'Dismount-Volume'  # Unmount volumes
    'fdisk' = 'Get-Disk'  # Disk management
    
    # System logging
    'dmesg' = 'Get-EventLog -LogName System -Newest 50'  # System logs
    'journalctl' = 'Get-WinEvent -FilterHashtable @{LogName="Application"} -MaxEvents 50'  # Application logs
    
    # User environment
    'env' = 'Get-ChildItem Env:'  # Environment variables
    'printenv' = 'Get-ChildItem Env:'  # Alternative to env
    
    # Process monitoring (advanced)
    'htop' = 'Get-Process | Sort-Object -Property CPU -Descending | Select-Object -First 20 ProcessName, CPU, WorkingSet, Id | Format-Table -AutoSize'
    'glances' = 'Get-Process | Sort-Object -Property CPU -Descending | Select-Object -First 20 ProcessName, CPU, WorkingSet, Id | Format-Table -AutoSize'
    
    # Network monitoring
    'iftop' = 'Get-NetAdapterStatistics | Select-Object Name, ReceivedBytes, SentBytes, ReceivedPacketsUnicastPerSecond, SentPacketsUnicastPerSecond'
    'nethogs' = 'Get-NetAdapterStatistics | Select-Object Name, ReceivedBytes, SentBytes, ReceivedPacketsUnicastPerSecond, SentPacketsUnicastPerSecond'
    
    # System information (detailed)
    'inxi' = 'systeminfo'  # Basic system information
    'neofetch' = 'systeminfo | Select-Object -First 20'  # System information (simplified)
    
    # File permissions (detailed)
    'getfacl' = 'Get-Acl'  # Get file/directory ACLs
    'setfacl' = 'Set-Acl'  # Set file/directory ACLs
    
    # Archive formats
    'gzip' = 'Compress-Archive'  # Basic compression
    'gunzip' = 'Expand-Archive'  # Basic extraction
    'bzip2' = 'Compress-Archive -CompressionLevel Optimal'  # Better compression
    'xz' = 'Compress-Archive -CompressionLevel Optimal'  # Best compression
    
    # System monitoring (advanced)
    'iotop' = 'Get-Process | Sort-Object -Property IO -Descending | Select-Object -First 10 ProcessName, CPU, IO'
    'nmon' = 'Get-Counter -ListSet * | Select-Object -ExpandProperty CounterSetName'  # Performance monitoring
    
    # Network tools
    'netcat' = 'Test-NetConnection'  # Basic network testing
    'telnet' = 'Test-NetConnection -Port'  # Port testing
    'traceroute' = 'Test-NetConnection -TraceRoute'  # Network path
    
    # System administration (advanced)
    'useradd' = 'New-LocalUser'  # User management
    'usermod' = 'Set-LocalUser'  # Modify user
    'userdel' = 'Remove-LocalUser'  # Delete user
    'groupadd' = 'New-LocalGroup'  # Group management
    'groupmod' = 'Rename-LocalGroup'  # Modify group
    'groupdel' = 'Remove-LocalGroup'  # Delete group
    
    # File system (advanced)
    'e2fsck' = 'chkdsk'  # Disk checking
    'fsck' = 'chkdsk'  # File system check
    'badblocks' = 'chkdsk /r'  # Bad sector checking
    
    # System time
    'ntpdate' = 'w32tm /resync'  # Time synchronization
    'hwclock' = 'Get-Date'  # Hardware clock
    
    # System logging (advanced)
    'syslog' = 'Get-EventLog -LogName System'  # System logs
    'logrotate' = 'wevtutil'  # Log rotation
    
    # System monitoring (processes)
    'pidof' = 'Get-Process -Name | Select-Object -ExpandProperty Id'  # Find process ID
    'pstree' = 'Get-Process | Format-Tree'  # Process tree
    
    # Network configuration
    'ifup' = 'Enable-NetAdapter'  # Bring interface up
    'ifdown' = 'Disable-NetAdapter'  # Bring interface down
    'ip' = 'Get-NetIPConfiguration'  # Network configuration
    
    # System information (hardware)
    'lshw' = 'Get-WmiObject -Class Win32_ComputerSystem'  # Hardware information
    'lscpu' = 'Get-WmiObject -Class Win32_Processor'  # CPU information
    'lsmem' = 'Get-WmiObject -Class Win32_PhysicalMemory'  # Memory information
    
    # System administration (services)
    'service' = 'Get-Service'  # Service management
    'systemctl' = 'Get-Service'  # Systemd service management
    
    # File system (permissions)
    'chattr' = 'Set-ItemProperty'  # Change file attributes
    'lsattr' = 'Get-ItemProperty'  # List file attributes
    
    # System monitoring (I/O)
    'iostat' = 'Get-Counter "\PhysicalDisk(*)\% Idle Time"'  # Disk I/O statistics
    'vmstat' = 'Get-Counter "\Memory\*" | Select-Object -ExpandProperty CounterSamples'  # Virtual memory statistics
    
    # Network monitoring (advanced)
    'tcpdump' = 'Get-NetEventSession'  # Packet capture (requires additional setup)
    'wireshark' = '& "C:\Program Files\Wireshark\Wireshark.exe"'  # If installed
    
    # System information (kernel)
    'uname -a' = 'systeminfo | Select-Object -First 1'  # System information
    'lsb_release' = 'Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion"'  # OS version
    
    # User environment (advanced)
    'su -' = 'Start-Process powershell -Verb RunAs'  # Switch user (admin)
    'sudo -i' = 'Start-Process powershell -Verb RunAs'  # Run as admin
    
    # Process management (signals)
    'kill -9' = 'Stop-Process -Id -Force'  # Force kill
    'pkill -f' = 'Get-Process | Where-Object { $_.ProcessName -like "*pattern*" } | Stop-Process -Force'  # Kill by pattern
    
    # File system (searching)
    'locate' = 'Get-ChildItem -Recurse -Filter'  # File search
    'updatedb' = 'Write-Host "Not needed on Windows"'  # Update file database
    
    # System administration (packages)
    'dpkg' = 'Get-Package'  # Package management
    'rpm' = 'Get-Package'  # Package management
    
    # System monitoring (resources)
    'free -m' = 'Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory, TotalVirtualMemorySize, FreeVirtualMemory | ForEach-Object { [PSCustomObject]@{ TotalMB = [math]::Round($_.TotalVisibleMemorySize/1KB, 2); UsedMB = [math]::Round(($_.TotalVisibleMemorySize - $_.FreePhysicalMemory)/1KB, 2); FreeMB = [math]::Round($_.FreePhysicalMemory/1KB, 2) } }'
    'df -h' = 'Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name="UsedGB";Expression={[math]::Round($_.Used/1GB, 2)}}, @{Name="FreeGB";Expression={[math]::Round($_.Free/1GB, 2)}}, @{Name="TotalGB";Expression={[math]::Round(($_.Used + $_.Free)/1GB, 2)}} | Format-Table -AutoSize'
    
    # System administration (users)
    'who -a' = 'Get-Process -IncludeUserName | Select-Object ProcessName, UserName -Unique'
    'w' = 'Get-Process -IncludeUserName | Select-Object ProcessName, UserName, CPU -First 10'
    'last' = 'Get-EventLog -LogName Security -InstanceId 4624, 4625 -After (Get-Date).AddDays(-7) | Select-Object TimeGenerated, Message'
    
    # File transfer (advanced)
    'scp' = 'Copy-Item -Path "source" -Destination "user@server:path" -ToSession (New-PSSession -ComputerName server)'
    'rsync' = 'Robocopy'  # Or use third-party rsync for Windows
    
    # System services (advanced)
    'systemctl' = 'Get-Service | Where-Object { $_.Status -eq "Running" }'  # List running services
    'service' = 'Get-Service'  # Service management
    
    # Package management (Windows specific)
    'choco' = 'choco'  # Chocolatey package manager
    'scoop' = 'scoop'  # Scoop package manager
    'winget' = 'winget'  # Windows Package Manager
    
    # Development tools (advanced)
    'make' = 'nmake'  # Or use WSL for full make functionality
    'gcc' = 'cl.exe'  # MSVC compiler
    'g++' = 'cl.exe'  # MSVC C++ compiler
    
    # Version control (advanced)
    'git' = 'git'  # Git is available cross-platform
    'svn' = 'svn'  # Subversion
    'hg' = 'hg'    # Mercurial
    
    # Text editors (advanced)
    'nano' = 'notepad'  # Basic text editing
    'vim' = 'vim'  # If Vim is installed
    'emacs' = 'emacs'  # If Emacs is installed
    
    # Shell and environment (advanced)
    'alias' = 'Set-Alias'  # Or use function for more complex aliases
    'history' = 'Get-History'  # Command history
    'type' = 'Get-Content'  # View file contents
    'touch' = 'New-Item -ItemType File -Force'  # Create empty file
    
    # File comparison (advanced)
    'diff' = 'Compare-Object'  # Or use git diff if available
    'cmp' = 'Compare-Object'  # File comparison
    
    # File permissions (Windows specific, advanced)
    'stat' = 'Get-ItemProperty'  # File status
    'file' = 'Get-Item'  # File type information
    
    # System information (advanced, detailed)
    'lspci' = 'Get-PnpDevice | Where-Object { $_.Class -eq "System devices" } | Select-Object Name, Status'
    'lsusb' = 'Get-PnpDevice | Where-Object { $_.Class -eq "USB" } | Select-Object Name, Status'

begin {
    # Initialize counters
    $script:issuesFound = 0
    $script:filesScanned = 0
    $script:totalIssues = 0
    $script:startTime = Get-Date
    
    # Output header
    Write-Host "`n=== Windows Compliance Check ===" -ForegroundColor Cyan
    Write-Host "Start Time: $($script:startTime)"
    Write-Host "Scanning path: $Path"
    Write-Host "File types: $($FileTypes -join ', ')"
    Write-Host "`nScanning files..."
}

process {
    try {
        $files = Get-ChildItem -Path $Path -File -Recurse -ErrorAction Stop | 
                Where-Object { $FileTypes -contains $_.Extension }
        
        foreach ($file in $files) {
            $script:filesScanned++
            $fileIssues = 0
            $relativePath = $file.FullName.Replace((Get-Location).Path, '').TrimStart('\')
            
            try {
                $content = Get-Content -Path $file.FullName -Raw -ErrorAction Stop
                $lineNumber = 0
                
                $content -split "`r?`n" | ForEach-Object {
                    $lineNumber++
                    $line = $_.Trim()
                    
                    # Skip empty lines and comments
                    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) { 
                        return 
                    }
                    
                    # Check for Linux/Unix commands
                    foreach ($linuxCmd in $script:commandMappings.Keys) {
                        if ($line -match "\b$([regex]::Escape($linuxCmd))\b") {
                            $script:issuesFound++
                            $fileIssues++
                            $script:totalIssues++
                            # Only show the first 5 issues per file to avoid flooding the output
                            if ($fileIssues -le 5) {
                                Write-Host "`n[!] Issue found in: $relativePath" -ForegroundColor Red
                                Write-Host "   Line $lineNumber`: " -NoNewline
                                Write-Host " $line" -ForegroundColor Yellow
                                Write-Host "   Suggestion: Use " -NoNewline
                                Write-Host "'$($script:commandMappings[$linuxCmd])'" -ForegroundColor Green -NoNewline
                                Write-Host " instead of " -NoNewline
                                Write-Host "'$linuxCmd'" -ForegroundColor Red
                                Write-Host ""
                            } elseif ($fileIssues -eq 6) {
                                Write-Host "   ... and $(($script:totalIssues - 5)) more issues in this file" -ForegroundColor DarkGray
                            }
                        }
                    }
                }
            } catch {
                Write-Warning "Error reading file $relativePath`: $_"
                continue
            }
            
            # Show progress
            if (($script:filesScanned % 10) -eq 0) {
                $percentComplete = [math]::Min(100, [math]::Floor(($script:filesScanned / 100) * 100))
                Write-Progress -Activity "Scanning files..." -Status "$script:filesScanned files scanned, $script:totalIssues issues found" -PercentComplete $percentComplete
            }
        }
    } catch {
        Write-Error "Error scanning directory: $_"
        return
    }
}

end {
    # Calculate statistics
    $endTime = Get-Date
    $duration = $endTime - $script:startTime
    $avgTimePerFile = if ($script:filesScanned -gt 0) { $duration.TotalMilliseconds / $script:filesScanned } else { 0 }
    
    # Output summary
    Write-Host "`n=== Scan Complete ===" -ForegroundColor Cyan
    Write-Host "Files scanned: $script:filesScanned"
    Write-Host "Total issues found: $script:totalIssues"
    Write-Host "Time taken: $($duration.TotalSeconds.ToString('0.00')) seconds"
    Write-Host "Average time per file: $($avgTimePerFile.ToString('0.00')) ms"
    
    if ($script:totalIssues -eq 0) {
        Write-Host "`n✅ No Linux/Unix commands found in the scanned files." -ForegroundColor Green
    } else {
        Write-Host "`n⚠️  Found $script:totalIssues potential compatibility issues." -ForegroundColor Yellow
        Write-Host "   Review the suggestions above and replace Linux/Unix commands with their PowerShell equivalents." -ForegroundColor Yellow
        
        # Output the most common issues
        $commonIssues = $script:commandMappings.Keys | 
            Where-Object { $_ -notin @('&&', '`', '| xargs') } | 
            Select-Object @{Name='Command';Expression={$_}}, 
                         @{Name='PowerShell';Expression={$script:commandMappings[$_]}} |
            Sort-Object Command
        
        Write-Host "`nCommon Linux to PowerShell Mappings:" -ForegroundColor Cyan
        $commonIssues | Format-Table -AutoSize | Out-String -Stream | ForEach-Object {
            if ($_ -match 'Command\s+PowerShell') {
                Write-Host $_ -ForegroundColor Cyan
                Write-Host ('-' * 80) -ForegroundColor Cyan
            } else {
                Write-Host $_
            }
        }
        
        # Return non-zero exit code if running in a CI/CD pipeline
        if ($env:CI -eq 'true') {
            exit 1
        }
    }
    
    # Clean up
    Write-Progress -Activity "Scan complete" -Completed
}
