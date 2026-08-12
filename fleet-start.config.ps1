# Per-repo fleet start config for local-llm-mcp
# Edit ports/backend target here - start.ps1 is fleet-standard.
@{
    Name         = 'local-llm-mcp'
    BackendPort  = 10833
    FrontendPort = 10832
    HealthPath   = '/health'
    WebRoot      = 'D:\Dev\repos\local-llm-mcp\web_sota'
    Backend = @{
        Kind          = 'uvicorn'
        UvicornTarget = 'llm_mcp.server:app'
        Env           = @{ WEB_PORT = '10833' }
    }
    Frontend = @{
        Kind           = 'vite-npm'
        PackageManager = 'npm'
        PortEnvVar     = 'VITE_PORT'
        ApiTargetEnv   = 'VITE_API_TARGET'
    }
}
