# Windows Compliance Tools

This directory contains tools to enforce Windows/PowerShell coding standards across the project.

## Check-WindowsCompliance.ps1

A PowerShell script that scans files for Linux/Unix commands and suggests PowerShell alternatives.

### Usage

```powershell

# Check current directory


.\Check-WindowsCompliance.ps1

# Check specific directory
.\Check-WindowsCompliance.ps1 -Path "path\to\directory"






# Check specific file types
.\Check-WindowsCompliance.ps1 -FileTypes @('.ps1', '.bat', '.md')





```

### Common Linux to PowerShell Mappings

| Linux Command | PowerShell Equivalent | Notes |
|--------------|----------------------|-------|
| `&&` | `;` | Command chaining |
| `ls` | `Get-ChildItem` | |
| `cat` | `Get-Content` | |
| `grep` | `Select-String` | |
| `rm` | `Remove-Item` | |
| `mkdir -p` | `New-Item -ItemType Directory -Force` | |
| `cp` | `Copy-Item` | |
| `mv` | `Move-Item` | |
| `chmod` | `icacls` or `Set-Acl` | |
| `pwd` | `Get-Location` | `$PWD` also works |
| `echo` | `Write-Output` | |
| `| xargs` | `| ForEach-Object` | |

## Git Pre-commit Hook

A pre-commit hook that runs the compliance check before each commit.

### Installation

$11. Copy `pre-commit` to `.git/hooks/`

$11. Make it executable (on Windows, this is usually not needed)



### Manual Run

You can manually run the pre-commit check with:

```powershell
.\Check-WindowsCompliance.ps1 -Path "path\to\files"





```

## Best Practices

$11. Always use PowerShell cmdlets instead of Linux/Unix commands

$11. Use full parameter names (e.g., `-Recurse` instead of `-r`)


$11. Quote paths with spaces

$11. Use `;` for command chaining instead of `&&`


$11. Use `$PWD` instead of `pwd` in scripts

## CI/CD Integration

Add this to your CI/CD pipeline to ensure compliance:

```yaml

- name: Check for Linux commands


  shell: pwsh
  run: |
    .\tools\Check-WindowsCompliance.ps1
    if ($LASTEXITCODE -ne 0) { exit 1 }

```

