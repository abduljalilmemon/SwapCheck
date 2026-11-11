"""SwapCheck: An open-source evaluation harness for LLM agents.

Tagline: Is it safe to swap your agent's model?
"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"

__version__ = "0.1.0"
