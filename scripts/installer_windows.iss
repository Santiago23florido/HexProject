#define MyAppName "HexProject"
#define MyAppVersion "1.0"
#define MyAppPublisher "HexProject"
#define MyAppExeName "bin\\hex_ui.exe"
#define MyAppSrcDir GetEnv('HEXPROJECT_APPDIR')

[Setup]
AppId={{B9E98F45-0D7B-4B9E-8F6B-4D7B0C1D6A10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputBaseFilename=HexProject-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "{#MyAppSrcDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}\bin"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}\bin"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"
