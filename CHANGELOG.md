# Endringslogg

Formatet følger [Keep a Changelog](https://keepachangelog.com/no/1.0.0/), og prosjektet bruker
[SemVer](https://semver.org/lang/no/).

## [1.0.0] — 2026-09-22

Første utgave av Fluesjakk: en norsk videreutvikling av fly-chess, med nytt grensesnitt, norsk
språk og egen skrivebordsapp.

### Nytt

- **Helt nytt grensesnitt.** Ny visuell identitet: mørk palett med lime-aksent, glassaktige paneler,
  ny typografisk oppbygning og en ny layout der brettet er det sentrale og visualiseringen ligger
  som en stripe under. Ingen deler av utseendet er arvet fra originalprosjektet.
- **Nye brettbrikker.** Seks brikker tegnet for dette prosjektet, i en kantet stil som skiller seg
  tydelig fra originalen. Koordinatene har fått egne farger med kontrast kontrollert mot WCAG AA.
- **Norsk grensesnitt.** All tekst brukeren ser, er på norsk bokmål og skrevet for norsk
  sjakktradisjon — sjakk, sjakk matt, patt, remis, bondeforvandling, springer, løper, tårn. Teksten
  ligger samlet i `web/i18n.js`, og en egen test feiler hvis det dukker opp engelske ord i
  grensesnittet.
- **«Slik fungerer Fluesjakk»** — en egen side som forklarer connectomet, kodingen av brettet,
  nevronnettverket, hva som er biologisk fastsatt og hva som er trent, vanskelighetsgradene og
  forskjellen på Python- og nettleserversjonen.
- **Innstillinger** med lagring: lyd, animasjoner, lovlige trekk, siste trekk, koordinater,
  fluas kommentarer, flueaktivitet, fluas øyne, brettema og teknisk modus.
- **Teknisk modus** som viser modellnavn, antall nevroner og koblinger, kjøremotor, verdihodets tall,
  valgt trekk, policy-topp og beregningstid.
- **Feilsøkingsside** med menneskelige forklaringer og en teknisk logg som kan kopieres ut.
- **Førstegangs nedlasting av modellen** med framdrift, sjekksumkontroll og lokal hurtiglagring.
  Spillet virker uten internett etterpå. Se [README](README.md#modell) for hvorfor vektene ikke
  ligger i kodelageret.
- **Skrivebordsapp for Windows** (`desktop/`) bygget med Electron: `Fluesjakk.exe`,
  `Fluesjakk-Setup.exe` og en zip-utgave. Ingen Python, Node eller terminal nødvendig.
- **Egen logo og egen fluefigur**, tegnet for prosjektet.
- **GitHub Actions** for tester, paritet og utgivelser, og en egen workflow som bygger
  Windows-pakken.
- **Norsk dokumentasjon**: `README.md`, `CREDITS.md`, `web/README.md`, `CONTRIBUTING.md`,
  `SECURITY.md` og denne endringsloggen.
- `scripts/hent_modell.py` henter modellfilene med framdrift og sjekker sjekksummen.

### Endret

- Alle 271 nettlesertestene kjører nå på Windows. `findChrome()` leter etter Chrome og Edge på
  Windows og macOS i tillegg til Linux, og de dynamiske importene i `flybrain.test.mjs` og
  `flybrain-gpu.test.mjs` bruker `file://`-adresser, som Nodes ESM-laster krever på Windows.
- `web/engine/worker.js` tar imot en ferdig innlest modell (`{type:'load', header, buffer}`) i
  tillegg til en URL. Regnestykkene er urørt; begge veier går gjennom samme `parseArrays`.
- `flychess/export/web.py` og `flychess/export/testvectors.py` importerer `torch` og
  modellklassene lat. Da kan testvektorene og `numpy_forward` bygges med numpy alene — det er slik
  paritetstesten nå kan kjøres uten GPU.
- `web/eye.js` og `web/brainviz.js` har fått norsk tekst og den nye fargepaletten. Ingen matematikk
  er endret.
- `web/test/style.test.mjs` håndhever det nye designet: kontrastkravene og den responsive
  rekkefølgen er de samme, men testene leser de nye fargetokensene.
- `web/test/app.test.mjs` er skrevet om for det nye grensesnittet og dekker i tillegg
  innstillinger, teknisk modus, nedlastingsflyten og at grensesnittet er norsk.
- Lisensen beholder opphavsretten til den arvede koden, som MIT krever.

### Rettet

- **Vanskelighetsgradene virket ikke.** Grensesnittet sendte `larve`/`flue`/`superflue` til
  arbeideren, som kjenner dem som `larva`/`fly`/`superfly`. Alle tre falt derfor stille tilbake til
  «flue»-nivået, og superflue kjørte aldri Monte-Carlo-tresøk. Oversettelsen er nå eksplisitt i
  `VANSKELIGHETER[].motor`, og en egen test sjekker at hvert nivå faktisk driver riktig motormodus.
- «Nytt parti» ryddet ikke vekk vennelagsstillingen fra forrige runde.
- Teknisk modus viste tomme felt hvis det ble slått på midt i et parti, fordi tallene bare ble
  samlet inn når panelet var synlig.
- `gazeLine` skrev «sin egen kongen»; etter «egen» skal brikken stå i ubestemt form.
- Brettet kunne bli høyere enn vinduet på lave skjermer. Bredden tar nå hensyn til vindushøyden.
- `web/test/mcts`-testene feilet på Python 3.12 med torch 2.14 på en maskin uten CUDA, av to grunner
  som begge er rettet: `configs/default.yaml` pinner `device: cuda` mens den innebygde standarden
  spør maskinen (testen normaliserer nå for det, som den allerede gjorde for `tiny.yaml`), og
  MCTS-testen krevde at 12 simuleringer fant matt-trekket `Ra8#` med et tilfeldig vektet nettverk —
  der er trekket aldri besøkt i det hele tatt. Budsjettet er hevet til 64, som ligger godt over
  punktet der søket når trekket uansett initialisering.
- `node --test web/test/` krevde et glob-mønster; dokumentasjonen viser nå `web/test/*.mjs`.

### Arvet

Motoren, treningen, eksporten, nettlesermotoren og spesifikasjonen kommer fra
[fly-chess](https://github.com/cesp99/fly-chess) av Carlo Esposito (MIT). Connectomet er
FlyWire-konsortiets arbeid (CC BY-NC 4.0). Se [CREDITS.md](CREDITS.md) for en presis oversikt over
hva som er hva.
