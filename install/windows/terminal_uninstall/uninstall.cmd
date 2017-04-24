SET SCRIPT_HOME=%~dp0
cd %SCRIPT_HOME%
DPInst.exe /D /Q /U ftdiport.inf
DPInst.exe /D /Q /U ftdibus.inf

del %SYSTEMROOT%\System32\drivers\ftdibus.sys >nul 2>&1
del %SYSTEMROOT%\System32\drivers\ftser2k.sys >nul 2>&1
