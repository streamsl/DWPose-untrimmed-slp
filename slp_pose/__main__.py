"""`python -m slp_pose ...`: pin GPU numbering before anything can start CUDA, then run the CLI."""
import sys

from slp_pose.env import configure_process

configure_process()

from slp_pose.cli import main  # noqa: E402  (after configure_process on purpose)

sys.exit(main())
