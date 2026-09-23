# Tredjeparter, data og lisenser

Koden i dette prosjektet er MIT ([LICENSE](LICENSE)). Alt under her har andre vilkår, og er grunnen
til at modellvektene ikke ligger i kodelageret.

Se [CREDITS.md](CREDITS.md) for hvem som har laget hva.

---

## 1. FlyWire-connectomet og modellvektene — CC BY-NC 4.0

Connectomdataene er hentet fra FlyWire (Codex snapshot 783, <https://codex.flywire.ai>) og er
lisensiert under **Creative Commons Attribution-NonCommercial 4.0 International**.
Zenodo-speilet (10.5281/zenodo.10676866) er under CC BY 4.0.

Modellvektene som følger med spillet er avledet av disse dataene og står under samme lisens.

> **Dette betyr at spillet og alt som bygger på modellen kan brukes fritt til ikke-kommersielle
> formål med attribusjon — men ikke selges eller brukes kommersielt.**

Full lisens: [docs/LICENSE-CC-BY-NC-4.0.txt](docs/LICENSE-CC-BY-NC-4.0.txt)

**Hvorfor vektene ikke ligger i git:** de har en annen lisens enn koden, og å blande dem ville gjort
det uklart hva som gjelder for hva. De hentes derfor ned separat — av spilleren ved første oppstart,
eller av byggeprosessen for Windows-utgaven (`scripts/hent_modell.py`). Kilden er
<https://huggingface.co/cesp99/fly-chess>.

---

## 2. Sjakkregler

| Bibliotek | Lisens | Hvor |
|---|---|---|
| [chess.js](https://github.com/jhlywa/chess.js) av Jeff Hlywa | BSD-2-Clause | `web/vendor/chess.js`, følger med i nettleser- og Windows-utgaven |
| [python-chess](https://github.com/niklasf/python-chess) av Niklas Fiekas | GPL-3.0-or-later | Python-siden: trening, spilling, evaluering, eksport |

`python-chess` er GPL, og Python-pakken som importerer det, distribueres derfor ikke som del av
Windows-utgaven. Den Windows-utgaven inneholder bare nettleserkoden (MIT og BSD-2) pluss
modellvektene (CC BY-NC 4.0).

---

## 3. Treningsdata

[Lichess åpne database](https://database.lichess.org) — **CC0** (offentlig eiendom). Partiene som
modellen er trent på, er filtrert til spillere ratet 1800 eller høyere.

---

## 4. Programvare prosjektet bygger på

| Prosjekt | Lisens | Brukes til |
|---|---|---|
| [PyTorch](https://pytorch.org) | BSD-3-Clause | trening og eksport |
| NumPy, SciPy, pandas, pyarrow | BSD-3-Clause | data og tallregning |
| [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org) | MIT / BSD | dashbordet |
| [Electron](https://www.electronjs.org) | MIT | skrivebordsappen |
| [chess.js](https://github.com/jhlywa/chess.js) | BSD-2-Clause | sjakkregler i nettleseren |

Electron-pakken inneholder sine egne lisensfiler: `LICENSE.electron.txt` og
`LICENSES.chromium.html` i den utpakkede Windows-utgaven.

---

## 5. Egne verk

Følgende er laget for Fluesjakk av Oliver Lysø Hellevik og er dekket av MIT-lisensen:

- grensesnittet, layouten og designsystemet (`web/index.html`, `web/style.css`, `web/app.js`)
- brettet og de seks brikkene (`web/board.js`)
- logoen (`web/assets/logo.svg`) og fluefiguren (`web/assets/flue.svg`)
- app-ikonet (`desktop/build/icon.png`)
- den norske teksten (`web/i18n.js`, `web/slik.js`)
- skrivebordsappen (`desktop/`)
- all norsk dokumentasjon

Ingen grafiske ressurser er hentet fra originalprosjektet.
