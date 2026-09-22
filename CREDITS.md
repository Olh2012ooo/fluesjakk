# Fluesjakk

**Laget og videreutviklet av Oliver Lysø Hellevik**

---

## Utvikler

**Oliver Lysø Hellevik** — konsept, grensesnitt, norsk språk, visuell identitet, skrivebordsapp og
distribusjon av denne versjonen.

Denne versjonen er en videreutvikling av prosjektet [fly-chess](https://github.com/cesp99/fly-chess)
av Carlo Esposito. Det prosjektet la grunnlaget: motoren, treningen, eksportformatet og
nettlesermotoren. Det du holder i hånden nå, er en egen norsk distribusjon med nytt grensesnitt,
ny visuell identitet, norsk språk og en egen skrivebordsapp.

Hva som er laget her, og hva som er arvet, står presist i [Hva er arvet, og hva er nytt](#hva-er-arvet-og-hva-er-nytt)
nederst. Ingenting av dette er ment å fremstille andres arbeid som mitt eget.

---

## Teknisk grunnlag og eksterne komponenter

Fluesjakk står på andres forskning og åpen kildekode. Dette er de delene, og hva de bidrar med:

### Connectomet — FlyWire

Hjernen i spillet er koblingskartet fra FlyWire, en fullstendig rekonstruksjon av hjernen til en
voksen fruktflue (*Drosophila melanogaster*) laget fra elektronmikroskopibilder ved synapsenivå.
Dette er ikke modellert eller forenklet — hver kobling i nettverket finnes i den ekte flua.

- Dorkenwald, S. *et al.* **Neuronal wiring diagram of an adult brain.** *Nature* 634, 124–138 (2024).
  <https://doi.org/10.1038/s41586-024-07558-y>
- Schlegel, P. *et al.* **Whole-brain annotation and multi-connectome cell typing of *Drosophila*.**
  *Nature* 634, 139–152 (2024). <https://doi.org/10.1038/s41586-024-07686-5>

Dataene er hentet fra FlyWire Codex (snapshot 783, <https://codex.flywire.ai>) og er lisensiert under
**CC BY-NC 4.0**. Speilet på Zenodo (10.5281/zenodo.10676866) er under CC BY 4.0.

Æren for hjernen tilhører FlyWire-konsortiet og de mange hundre korrekturleserne som bygde
rekonstruksjonen over flere år.

### Nevrotransmittere

Fortegnet til hver kobling — om den virker stimulerende eller hemmende — kommer fra en
maskinlæringsmodell som klassifiserer signaltypen fra elektronmikroskopibildene av koblingspunktene:

- Eckstein, N. *et al.* **Neurotransmitter classification from electron microscopy images at synaptic
  sites in *Drosophila melanogaster*.** *Cell* 187 (2024).

Glutamat behandles som hemmende (GluCl), slik som er vanlig for insektsnervesystemet.

### Treningsdata

Modellen er trent på å etterligne sjakktrekk fra den åpne partidatabasen til Lichess, filtrert til
partier der begge spillere var ratet 1800 eller høyere.

- [Lichess åpne database](https://database.lichess.org) — **CC0** (offentlig eiendom)

### Sjakkregler

- **chess.js** av Jeff Hlywa — BSD-2-Clause. Brukes i nettleseren og i skrivebordsappen.
- **python-chess** av Niklas Fiekas — GPL-3.0-or-later. Brukes av Python-siden til trekkgenereing,
  trening, evaluering og eksport.

### Modellvektene

Vektene som gjør at flua faktisk kan spille, er avledet av FlyWire-dataene og er derfor underlagt
**CC BY-NC 4.0**. De kan brukes fritt til ikke-kommersielle formål med attribusjon, men ikke selges
eller brukes kommersielt.

Fordi vektene har en annen lisens enn koden, ligger de ikke i dette kodelageret. Spillet laster dem
ned første gang det startes, eller de legges ved siden av spillet ved bygging av en egen utgave.
Se [README](README.md#modell) for detaljer.

### Verktøy og biblioteker

Fluesjakk bruker PyTorch (BSD-3-Clause), NumPy/SciPy/pandas/pyarrow (BSD-3-Clause),
FastAPI og Uvicorn (MIT/BSD) og Electron (MIT). Ingen av dem er laget for dette prosjektet.

---

## Hva er arvet, og hva er nytt

Denne oversikten finnes fordi det skal være mulig å se nøyaktig hva som er hvem sitt.

### Arvet fra fly-chess (Carlo Esposito, MIT)

- Den numeriske modellen og framoverpasset: `flychess/model/`, `flychess/connectome/`
- Trening, selvspill og Monte-Carlo-tresøk: `flychess/train/`
- Sjakkmiljøet og kodingen av brett og trekk: `flychess/chessenv/`
- Eksporten til nettleserformatet og referanseimplementasjonen i numpy: `flychess/export/`
- Nettlesermotoren, Web Worker-protokollen og tverrspråkstesten: `web/engine/`
- Eksporttjenesten og den lokale tjeneren: `flychess/play/local_web.py`
- Spesifikasjonen: `docs/SPEC.md`, med `docs/DATA.md`, `docs/RETINA.md` og lisensfilen for CC BY-NC
- Skjermbildene `docs/dashboard.png`, `docs/dashboard-mobile.png` og `docs/retina-map.png` viser
  det arvede Python-dashboardet og retina-kartleggingen, og er hentet fra originalprosjektet.
  Skjermbildene i `docs/skjermbilder/` er av Fluesjakk og tatt her.

### Nytt i Fluesjakk (Oliver Lysø Hellevik)

- Alt brukergrensesnitt: `web/index.html`, `web/style.css`, `web/app.js`, `web/board.js`,
  `web/slik.js`, `web/i18n.js`
- Norsk språk i hele grensesnittet, omskrevet for norsk sjakktradisjon — ikke maskinoversatt
- Ny visuell identitet: ny palett, nye brettbrikker tegnet for dette prosjektet, ny logo, ny flyfigur
- Nye funksjoner: innstillinger med lagring, teknisk modus, «Slik fungerer Fluesjakk»,
  feilsøkingsside, førstegangs nedlasting av modellen med framdrift
- Norsk brukerdokumentasjon: `README.md`, `CREDITS.md`, `docs/`
- Skrivebordsappen for Windows: `desktop/`
- GitHub Actions for tester, paritet og utgivelser

### Endret i den arvede koden

Få og små endringer, alle for å gjøre tverspråkstesten kjørbar uten GPU og for å gjøre
nettlesermotoren tilgjengelig uten å hente modellen fra en URL:

- `flychess/export/web.py` og `flychess/export/testvectors.py`: `torch` og modellklassene
  importeres nå lat, slik at testvektorene kan bygges med numpy alene.
- `web/engine/worker.js`: tar imot en ferdig innlest modell i tillegg til en URL. Selve
  regnestykkene er urørt.
- `web/eye.js` og `web/brainviz.js`: norsk tekst og ny fargepalett. Ingen matematikk endret.

Ingen vekter, ingen koblinger, ingen trekkindekser og ingen arkitektur er endret.

---

## Sitering

Bruker du dette i forskning eller undervisning, siter FlyWire-artiklene over. Dette prosjektet er en
demonstrasjon bygget på deres arbeid, og krever ingen egen sitering.
