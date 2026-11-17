; Framewise NSIS hooks (Tauri `bundle.windows.nsis.installerHooks`).
; Makes `framewise` (and the short `fw`) work in any terminal without editing PATH: one-line
; launchers go into %LOCALAPPDATA%\Microsoft\WindowsApps, which Windows 10/11 already has on PATH.

!macro NSIS_HOOK_POSTINSTALL
  CreateDirectory "$LOCALAPPDATA\Microsoft\WindowsApps"
  FileOpen $0 "$LOCALAPPDATA\Microsoft\WindowsApps\framewise.cmd" w
  FileWrite $0 '@"$INSTDIR\framewise-engine.exe" cli %*$\r$\n'
  FileClose $0
  FileOpen $0 "$LOCALAPPDATA\Microsoft\WindowsApps\fw.cmd" w
  FileWrite $0 '@"$INSTDIR\framewise-engine.exe" cli %*$\r$\n'
  FileClose $0
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  Delete "$LOCALAPPDATA\Microsoft\WindowsApps\framewise.cmd"
  Delete "$LOCALAPPDATA\Microsoft\WindowsApps\fw.cmd"
!macroend
