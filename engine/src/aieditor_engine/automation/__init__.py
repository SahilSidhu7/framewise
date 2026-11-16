"""Automation: the job engine behind the public /v1 API, the CLI, watch folders and the MCP server.

Everything here runs on the user's own machine. A job is: input files + a recipe
(reel, highlights, captions, video_edit, photo_edit, render_program, transcribe)
+ the prompt options (style, output). It runs in a throwaway project under the
automation home folder and leaves its results in that job's `out/` folder.
"""

from .home import automation_home

__all__ = ["automation_home"]
