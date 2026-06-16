import os
import sys
import traceback
import asyncio
import subprocess
from pathlib import Path

# Auto-reexecute with .venv Python if not already using it
project_root = Path(__file__).resolve().parent
venv_python = project_root / ".venv" / "Scripts" / "python.exe"
if sys.platform == "win32" and venv_python.exists():
    real_venv = str(venv_python.resolve()).lower()
    real_curr = str(Path(sys.executable).resolve()).lower()
    if real_curr != real_venv:
        try:
            sys.exit(subprocess.call([str(venv_python.resolve())] + sys.argv))
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"Failed to auto-delegate to virtual environment: {e}")

# Add src to Python path
sys.path.append(str(project_root / "src"))

def write_crash_report(err_msg, exc=None):
    report_path = Path(__file__).resolve().parent / "crash_report.log"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CRASH DETECTED: {err_msg}\n")
        if exc:
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write("=" * 80 + "\n")
    print(f"\nCrash report written to {report_path}")

def custom_excepthook(type, value, tb):
    tb_msg = "".join(traceback.format_exception(type, value, tb))
    print("Unhandled Exception:", tb_msg, file=sys.stderr)
    write_crash_report(f"Unhandled thread exception: {value}", value)

# Set global excepthook for standard threads
sys.excepthook = custom_excepthook

def custom_asyncio_exception_handler(loop, context):
    exception = context.get("exception")
    message = context.get("message")
    err_msg = f"Asyncio exception: {message}"
    if exception:
        err_msg += f" ({exception})"
    print("Asyncio Error:", err_msg, file=sys.stderr)
    
    # Format and save traceback if available
    report_path = Path(__file__).resolve().parent / "crash_report.log"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CRASH DETECTED: {err_msg}\n")
        if exception:
            f.write("".join(traceback.format_exception(type(exception), exception, exception.__traceback__)))
        f.write("=" * 80 + "\n")

def main():
    try:
        from jarvis.main import main as jarvis_main
        # Initialize default event loop if needed to set exception handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(custom_asyncio_exception_handler)
        
        print("Running Jarvis...")
        jarvis_main()
    except Exception as e:
        print(f"Exception in main thread: {e}", file=sys.stderr)
        write_crash_report(f"Main thread exception: {e}", e)
        raise

if __name__ == "__main__":
    main()
