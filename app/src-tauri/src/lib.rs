//! Desktop shell: starts the Python engine as a child process, hands its
//! port + per-launch token to the web UI, and stops it on exit.
//!
//! Engine resolution order:
//! 1. `AIEDITOR_ENGINE_PYTHON` env var -> `<python> -m aieditor_engine`
//! 2. debug builds: `engine/.venv` next to this repo -> `python -m aieditor_engine`
//! 3. release builds: bundled sidecar `framewise-engine[.exe]` beside the app executable

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{Manager, RunEvent, State};

#[derive(Clone, Serialize, Deserialize)]
struct EngineInfo {
    port: u16,
    token: String,
}

#[derive(Default)]
struct Engine {
    child: Mutex<Option<Child>>,
    info: Mutex<Option<EngineInfo>>,
    error: Mutex<Option<String>>,
}

#[tauri::command]
fn engine_info(engine: State<'_, Engine>) -> Result<EngineInfo, String> {
    if let Some(info) = engine.info.lock().unwrap().clone() {
        return Ok(info);
    }
    Err(engine
        .error
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| "engine starting".into()))
}

fn engine_command() -> Result<Command, String> {
    if let Ok(python) = std::env::var("AIEDITOR_ENGINE_PYTHON") {
        let mut cmd = Command::new(python);
        cmd.args(["-m", "aieditor_engine"]);
        return Ok(cmd);
    }

    if cfg!(debug_assertions) {
        let engine_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine");
        let python = if cfg!(windows) {
            engine_dir.join(".venv/Scripts/python.exe")
        } else {
            engine_dir.join(".venv/bin/python")
        };
        if python.exists() {
            let mut cmd = Command::new(python);
            cmd.args(["-m", "aieditor_engine"]);
            return Ok(cmd);
        }
        return Err(format!(
            "dev engine not found at {}. Run `uv sync` in engine/ or set AIEDITOR_ENGINE_PYTHON.",
            engine_dir.display()
        ));
    }

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let name = if cfg!(windows) { "framewise-engine.exe" } else { "framewise-engine" };
    let sidecar = exe
        .parent()
        .ok_or("cannot locate app directory")?
        .join(name);
    if !sidecar.exists() {
        return Err(format!("engine binary missing: {}", sidecar.display()));
    }
    Ok(Command::new(sidecar))
}

fn start_engine(app: tauri::AppHandle) {
    std::thread::spawn(move || {
        let engine = app.state::<Engine>();
        let fail = |msg: String| *engine.error.lock().unwrap() = Some(msg);

        let mut cmd = match engine_command() {
            Ok(cmd) => cmd,
            Err(e) => return fail(e),
        };
        cmd.stdout(Stdio::piped()).stderr(Stdio::inherit()).stdin(Stdio::null());
        // The frozen engine is a launcher plus a child process; killing the launcher alone would
        // orphan the child. The engine watches this PID and exits when the app is gone.
        cmd.env("AIEDITOR_PARENT_PID", std::process::id().to_string());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return fail(format!("failed to start engine: {e}")),
        };
        let stdout = child.stdout.take().expect("piped stdout");
        *engine.child.lock().unwrap() = Some(child);

        // First stdout line is {"event":"ready","port":..,"token":..}; keep draining afterwards
        // so the pipe never fills and blocks the engine.
        let mut lines = BufReader::new(stdout).lines();
        match lines.next() {
            Some(Ok(line)) => match serde_json::from_str::<EngineInfo>(&line) {
                Ok(info) => *engine.info.lock().unwrap() = Some(info),
                Err(e) => return fail(format!("bad engine handshake ({e}): {line}")),
            },
            _ => return fail("engine exited before it was ready; see logs".into()),
        }
        for _ in lines.map_while(Result::ok) {}
    });
}

fn stop_engine(app: &tauri::AppHandle) {
    if let Some(mut child) = app.state::<Engine>().child.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Engine::default())
        .invoke_handler(tauri::generate_handler![engine_info])
        .setup(|app| {
            start_engine(app.handle().clone());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Framewise")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop_engine(app);
            }
        });
}
