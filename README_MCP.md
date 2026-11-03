# PhenoAssistant – MCP Branch

All 25 tools now connect to the assistant through one MCP server. To add a tool, you add one line to a list.

**Key files**
- `tool_registry.py` – the list of tools. The only file you edit.
- `server.py` – the MCP server. Built from the list automatically.
- `bridge_factory.py` – connects the assistant to the server.
- `smoke_test.py` – lists all available tools.
- `call_tool.py` – runs one tool without the AI.
- `client.py`, `mcp_bridge.py` – old prototypes. Not used.

---

## Setup

1. Follow the setup in the main `README.md`.
2. Switch to this branch: `git checkout mcp`
3. Create `.env.yaml` in the project folder:
   ```yaml
   API_SUPPLIER: openrouter
   MODEL_NAME: openai/gpt-4o
   OPENAI_API_KEY: your-key-here
   ```
4. Run `conda activate phenoassistant` in each new terminal.

---

## 1. Test that MCP works

**Check 1 – Are all the tools loaded?** (no API key needed)
```bash
python smoke_test.py
```
- Expected: `25 tool(s) mounted`.

**Check 2 – Does a tool work through MCP?** (no API key needed)
```bash
python call_tool.py perform_anova '{"data_path": "data_for_eval/aracrop_phenotypes.csv", "descriptor": "average_leaf_area", "within_subject_factor": "days_after_sowing", "between_subject_factor": "ecotype", "subject_id": "plant_id"}'
```
- Expected: `=== RESULT ===` followed by an ANOVA table.

**Check 3 – Full case study** (needs a GPU and an API key)
- Open `case1.ipynb`, `case2.ipynb` or `case3.ipynb` and run all cells.
- No changes are needed. Tools go through MCP automatically.
- Tool outputs (counts, areas, statistics) should match the saved results.
- The AI's final summary may differ between runs. Trust the tool outputs.
- If it seems frozen, press Enter at `Replying as Admin...`.

---

## 2. Add a new tool

1. Write a function in `functions/`, e.g. `functions/my_tool.py`:
   ```python
   from typing import Annotated

   def count_leaves(img_path: Annotated[str, "Path to the plant image"]) -> str:
       ...
       return "6"
   ```
2. In `tool_registry.py`, import it and add one line to `TOOL_REGISTRY`:
   ```python
   from functions.my_tool import count_leaves

   (count_leaves, "count_leaves", "Count the leaves in a plant image."),
   ```
   - Name: must be unique.
   - Description: tells the AI when to use the tool. Keep it clear.
3. Run `python smoke_test.py`. Expected: `26 tool(s) mounted`.

> ⚠️ Tools added with `@pheno_tool` (in `functions/new_tools.py`) still work but **bypass MCP**. Use `tool_registry.py` instead.

---

## 3. Servers and user interfaces

**Does MCP create a server automatically?**
- Yes. `server.py` is generated from the tool list.
- It is not a permanently running service. The assistant starts it when needed and closes it after.
- You never start it manually.

**What should a user interface call?**
- **Chat UI** (user types requests in plain English): call the assistant, i.e. the `manager` in `agents.py`. See `chat.py` for a minimal example. **Recommended.**
- **Tool-button UI** (one button runs one tool): call the MCP server. To make it reachable over a network, change the last line of `server.py` to:
  ```python
  mcp.run(transport="streamable-http")
  ```
  > ⚠️ **Untested** on this branch.
