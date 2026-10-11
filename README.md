# jentak – generovani obrazku + videa na Molab (RTX Pro 6000)

## Postup

1. Zkopiruj `.env.example` do `.env` (hotovo) a dopln tokeny:
   - `HF_TOKEN` – https://huggingface.co/settings/tokens (staci read).
     Na https://huggingface.co/Lightricks/LTX-2.5 klikni **Agree and Access**.
   - `GITHUB_TOKEN` – https://github.com/settings/tokens (scope `repo`).
2. Rekni agentovi: `python3 scripts/check_env.py`
3. Agent zalozí GitHub repo, pushne kostru a napojí Molab synced notebook.
4. Tokeny se nikam necommitují (`.env` je v `.gitignore`) a do notebooku
   se nedávají – do session se nastavují jen docasne pres pair.

## Stack

- Obrazky: `Qwen/Qwen-Image-2.1` (diffusers)
- Video: `Lightricks/LTX-2.5` distilled (diffusers / ltx-pipelines)
- Prostredi: Molab marimo + RTX Pro 6000 Blackwell 96 GB
