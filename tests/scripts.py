import subprocess

subprocess.Popen(
    [
        "gnome-terminal",
        "--",
        "bash",
        "-c",
        """
        uv run main.py --text --script scripts/genesis.json
        exec bash
        """
    ]
)