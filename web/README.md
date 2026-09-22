# web/ — Fluesjakk i nettleseren

En statisk side uten byggetrinn og uten CDN. Den kjører nøyaktig det samme nettverket som Python
trente opp: FlyWire-connectomet som et rekurrent nett, evaluert inne i en **Web Worker** — på
skjermkortet gjennom WebGPU når nettleseren har det, ellers med håndskrevne sparsomme
typed-array-kjerner. Det er den samme matematikken, og fallback skjer automatisk.

## Filer

```
index.html            skjermbildene: forside, parti, «Slik fungerer det»
style.css             designsystemet: fargetokens, layout, brettet, responsivt
app.js                appkontrolleren: spillflyt, vennelag, innstillinger, nedlasting av modellen
i18n.js               ALL tekst brukeren ser, på norsk bokmål
slik.js               innholdet på «Slik fungerer det» og i feilsøkingsdialogen
board.js              sjakkbrettet i SVG (dra / klikk, lovlige trekk, bondeforvandling)
eye.js                «Fluas øyne»: begge fasettøynene fra modellens retina-kart
brainviz.js           nevronaktiviteten og tankegjenspillingen gjennom tidsstegene
assets/               egen logo og egen fluefigur
engine/               motoren — se under
vendor/chess.js       chess.js 1.4.0 (BSD-2)
model/                utelatt fra git: brain.json + brain.flyb + brain.flyb.gz
test/                 node --test
```

## Motoren

Motoren er den delen som må stå urørt, fordi den er pinnet mot Python av paritetstesten.

```
engine/loader.js         henter brain.json + brain.flyb(.gz), pakker ut, hurtiglager, tolker (§8);
                         modelFeatures(): hvilke valgfrie funksjoner en blob har
engine/flybrain.js       FlyBrain.forward(): SPEC §4 i ren JavaScript
engine/flybrain-gpu.js   det samme som WGSL-compute-shadere, pluss f32-speil for test
engine/mcts.js           PUCT Monte-Carlo-tresøk for superflue
engine/worker.js         Web Worker-protokollen, vanskelighetsgradene, valg av motor
engine/encoding.js       brett → plan, trekk ↔ indeks (delt med Python)
```

`worker.js` tar imot modellen på to måter: fra en URL (`{type:'load', baseUrl}`) eller ferdig
innlest fra siden (`{type:'load', header, buffer}`). Den andre veien brukes når spillet har lastet
ned modellen selv og lagt den i sitt eget hurtiglager. Begge veier går gjennom nøyaktig samme
`parseArrays`, så tallene motoren regner på, er identiske uansett hvordan bytene kom inn.

## Protokollen mellom siden og motoren

```
→ {type:'load', baseUrl}                        eller {type:'load', header, buffer, fromCache}
← {type:'progress', loaded, total, phase, n, nnz, runName}
← {type:'ready', header, legend, features, retina, sample, silhouette, fromCache, bytes, backend, gpu}
→ {type:'move', id, fen, moves:[uci…], difficulty, trace?}   difficulty: 'larva' | 'fly' | 'superfly'
← {type:'thinking', id, done, total}            bare superflue, hver tiende simulering
← {type:'move', id, move, san, policyTop, value, activitySample, retinaDrive, trace, thinkMs, sims, backend}
→ {type:'eval', id, fen, moves}
← {type:'eval', id, value, policyTop, activitySample, retinaDrive, trace, traceSteps, backend}
→ {type:'cancel', id}                           avbryt et pågående superflue-søk
← {type:'backend', backend:'js', reason}        uoppfordret: WebGPU-enheten ble borte
← {type:'error', id?, message}
```

`moves` er hele trekkhistorikken fra startstillingen, slik at gjentakelsesregelen virker. `value` er
alltid fra den som skal trekke sitt synspunkt — altså fluas eget syn når det er fluas tur.

## Vanskelighetsgrader

Nøklene i protokollen er engelske fordi de er en del av kontrakten mot `worker.js` (SPEC §9). Det
norske grensesnittet oversetter dem i `i18n.js` (`VANSKELIGHETER[].motor`), og
`web/test/app.test.mjs` sjekker at hvert nivå faktisk driver riktig motormodus — en feil her ville
ellers vært usynlig, siden alle nivåene ville sett like ut i grensesnittet.

| Norsk | Protokoll | Hvordan trekket velges |
|---|---|---|
| Larve | `larva` | trekker fra policyen med temperatur 1,2, uten søk |
| Flue | `fly` | de tre beste policy-trekkene prøves ett steg fram; verdihodet avgjør |
| Superflue | `superfly` | PUCT-tresøk, 400 simuleringer på WebGPU og 40 i ren JavaScript |

## Legge til et nytt språk

All tekst brukeren ser, ligger i `i18n.js`. En engelsk versjon er en ny ordbok med samme nøkler,
pluss `VANSKELIGHETER` med oversatte navn. Ingen andre filer trenger å endres — bortsett fra
`web/test/app.test.mjs`, som med vilje feiler hvis det dukker opp engelske ord i grensesnittet.

## Kjøre lokalt

```bash
fly export-web --run <kjøring>     # skriver web/model/
python -m http.server 8000 --directory web
```

Uten `web/model/` tilbyr siden å laste modellen ned fra Hugging Face i stedet.

## Tester

```bash
node --test web/test/*.mjs
```

| Fil | Dekker |
|---|---|
| `parity.test.mjs` | begge motorene mot Pythons referanse på den eksporterte modellen (SPEC §8) |
| `flybrain.test.mjs` | framoverpasset mot en float64-referanse, MCTS på en stubbehjerne |
| `flybrain-gpu.test.mjs` | WGSL-kjernene gjennom sine f32-speil, asynkron MCTS |
| `features.test.mjs` | de valgfrie funksjonene i §8 i begge motorene |
| `encoding.test.mjs` | brettkodingen mot `tests/vectors/encoding.json` |
| `loader.test.mjs` | henting, versjonering, sjekksum og hurtiglager |
| `board.test.mjs` | brettets tilstandsmaskin: forvandlingsvelgeren, tastaturnavigasjon |
| `eye.test.mjs` | de rene hjelpefunksjonene i `eye.js` og `brainviz.js` |
| `style.test.mjs` | kontrast (WCAG) og den responsive rekkefølgen i `style.css` |
| `app.test.mjs` | hele siden i headless Chromium: spillflyt, vennelag, feilhåndtering, norsk språk |

Paritetstesten hopper over med en melding når `tests/vectors/model.json` eller `web/model/` mangler.
