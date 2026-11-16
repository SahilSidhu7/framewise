; AIEditor NSIS hooks (Tauri `bundle.windows.nsis.installerHooks`).
; Makes `aieditor` work in any terminal without editing PATH: a one-line launcher goes into
; %LOCALAPPDATA%\Microsoft\WindowsApps, which Windows 10/11 already has on the user's PATH.

!macro NSIS_HOOK_POSTINSTALL
  CreateDirectory "$LOCALAPPDATA\Microsoft\WindowsApps"
  FileOpen $0 "$LOCALAPPDATA\Microsoft\WindowsApps\aieditor.cmd" w
  FileWrite $0 '@"$INSTDIR\aieditor-engine.exe" cli %*$\r$\n'
  FileClose $0
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  Delete "$LOCALAPPDATA\Microsoft\WindowsApps\aieditor.cmd"
!macroend
